"""
GST Protocol objects
"""
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
from pygsti.protocols.estimate import Estimate as _Estimate
from ..objects.circuitstructure import PlaquetteGridCircuitStructure as _PlaquetteGridCircuitStructure
from ..objects.gaugegroup import TrivialGaugeGroup as _TrivialGaugeGroup
from ..objects.gaugegroup import TrivialGaugeGroupElement as _TrivialGaugeGroupElement
from ..objects.circuitlist import CircuitList as _CircuitList
from ..objects.resourceallocation import ResourceAllocation as _ResourceAllocation
from ..objects.termforwardsim import TermForwardSimulator as _TermFSim
from ..objects.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore


#For results object:


ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0


class HasTargetModel(object):
    """
    Adds to an experiment design a target model

    Parameters
    ----------
    target_model_filename_or_obj : Model or str
        Target model or path to a file containing a model.
    """

    def __init__(self, target_model_filename_or_obj):
        self.target_model = _load_model(target_model_filename_or_obj)
        self.auxfile_types['target_model'] = 'pickle'


class GateSetTomographyDesign(_proto.CircuitListsDesign, HasTargetModel):
    """
    Minimal experiment design needed for GST

    Parameters
    ----------
    target_model_filename_or_obj : Model or str
        Target model or the path to a file containing the target model.

    circuit_lists : list
        Per-GST-iteration circuit lists, giving the circuits to run at each GST
        iteration (typically these correspond to different maximum-lengths).

    all_circuits_needing_data : list, optional
        A list of all the circuits in `circuit_lists` typically with duplicates removed.

    qubit_labels : tuple, optional
        The qubits that this experiment design applies to. If None, the line labels
        of the first circuit is used.

    nested : bool, optional
        Whether the elements of `circuit_lists` are nested, e.g. whether
        `circuit_lists[i]` is a subset of `circuit_lists[i+1]`.  This
        is useful to know because certain operations can be more efficient
        when it is known that the lists are nested.

    remove_duplicates : bool, optional
        Whether to remove duplicates when automatically creating
        all the circuits that need data (this argument isn't used
        when `all_circuits_needing_data` is given).
    """

    def __init__(self, target_model_filename_or_obj, circuit_lists, all_circuits_needing_data=None,
                 qubit_labels=None, nested=False, remove_duplicates=True):
        super().__init__(circuit_lists, all_circuits_needing_data, qubit_labels, nested, remove_duplicates)
        HasTargetModel.__init__(self, target_model_filename_or_obj)


class StandardGSTDesign(GateSetTomographyDesign):
    """
    Standard GST experiment design consisting of germ-powers sandwiched between fiducials.

    Parameters
    ----------
    target_model_filename_or_obj : Model or str
        Target model or the path to a file containing the target model.

    prep_fiducial_list_or_filename : list or str
        A list of preparation fiducial :class:`Circuit`s or the path to a filename containing them.

    meas_fiducial_list_or_filename : list or str
        A list of measurement fiducial :class:`Circuit`s or the path to a filename containing them.

    germ_list_or_filename : list or str
        A list of germ :class:`Circuit`s or the path to a filename containing them.

    max_lengths : list
        List of integers, one per LSGST iteration, which set truncation lengths
        for repeated germ strings.  The list of circuits for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    germ_length_limits : dict, optional
        A dictionary limiting the max-length values used for specific germs.
        Keys are germ sequences and values are integers.  For example, if
        this argument is `{('Gx',): 4}` and `max_length_list = [1,2,4,8,16]`,
        then the germ `('Gx',)` is only repeated using max-lengths of 1, 2,
        and 4 (whereas other germs use all the values in `max_length_list`).

    fiducial_pairs : list of 2-tuples or dict, optional
        Specifies a subset of all fiducial string pairs (prepStr, effectStr)
        to be used in the circuit lists.  If a list, each element of
        fid_pairs is a (iPrepStr, iEffectStr) 2-tuple of integers, each
        indexing a string within prep_strs and effect_strs, respectively, so
        that prepStr = prep_strs[iPrepStr] and effectStr =
        effect_strs[iEffectStr].  If a dictionary, keys are germs (elements
        of germ_list) and values are lists of 2-tuples specifying the pairs
        to use for that germ.

    keep_fraction : float, optional
        The fraction of fiducial pairs selected for each germ-power base
        string.  The default includes all fiducial pairs.  Note that
        for each germ-power the selected pairs are *different* random
        sets of all possible pairs (unlike fid_pairs, which specifies the
        *same* fiducial pairs for *all* same-germ base strings).  If
        fid_pairs is used in conjuction with keep_fraction, the pairs
        specified by fid_pairs are always selected, and any additional
        pairs are randomly selected.

    keep_seed : int, optional
        The seed used for random fiducial pair selection (only relevant
        when keep_fraction < 1).

    include_lgst : boolean, optional
        If true, then the starting list (only applicable when
        `nest == True`) is the list of LGST strings rather than the
        empty list.  This means that when `nest == True`, the LGST
        sequences will be included in all the lists.

    nest : boolean, optional
        If True, the GST circuit lists are "nested", meaning
        that each successive list of circuits contains all the gate
        strings found in previous lists (and usually some additional
        new ones).  If False, then the returned circuit list for maximum
        length == L contains *only* those circuits specified in the
        description above, and *not* those for previous values of L.

    circuit_rules : list, optional
        A list of `(find,replace)` 2-tuples which specify circuit-label replacement
        rules.  Both `find` and `replace` are tuples of operation labels (or `Circuit` objects).

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset.  This information is stored within the returned circuit
        structures.  Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    dscheck : DataSet, optional
        A data set which filters the circuits used for GST. When a standard-GST
        circuit is missing from this `DataSet`, action is taken according to
        `action_if_missing`.

    action_if_missing : {"raise","drop"}, optional
        The action to take when a desired circuit is missing from
        `dscheck` (only relevant when `dscheck` is not None).  "raise" causes
        a ValueError to be raised; "drop" causes the missing sequences to be
        dropped from the returned set.

    qubit_labels : tuple, optional
        The qubits that this experiment design applies to. If None, the line labels
        of the first circuit is used.

    verbosity : int, optional
        The level of output to print to stdout.

    add_default_protocol : bool, optional
        Whether a default :class:`StandardGST` protocol should be added to this
        experiment design.  Setting this to True makes it easy to analyze the data
        (after it's gathered) corresponding to this design via a :class:`DefaultRunner`.
    """

    def __init__(self, target_model_filename_or_obj, prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
                 germ_list_or_filename, max_lengths, germ_length_limits=None, fiducial_pairs=None, keep_fraction=1,
                 keep_seed=None, include_lgst=True, nest=True, circuit_rules=None, op_label_aliases=None,
                 dscheck=None, action_if_missing="raise", qubit_labels=None, verbosity=0,
                 add_default_protocol=False):

        #Get/load fiducials and germs
        prep, meas, germs = _load_fiducials_and_germs(
            prep_fiducial_list_or_filename,
            meas_fiducial_list_or_filename,
            germ_list_or_filename)
        self.prep_fiducials = prep
        self.meas_fiducials = meas
        self.germs = germs
        self.maxlengths = max_lengths
        self.germ_length_limits = germ_length_limits
        self.include_lgst = include_lgst
        self.aliases = op_label_aliases
        self.circuit_rules = circuit_rules

        #Hardcoded for now... - include so gets written when serialized
        self.truncation_method = "whole germ powers"
        self.nested = nest

        #FPR support
        self.fiducial_pairs = fiducial_pairs
        self.fpr_keep_fraction = keep_fraction
        self.fpr_keep_seed = keep_seed

        #TODO: add a line_labels arg to create_lsgst_circuit_lists and pass qubit_labels in?
        target_model = _load_model(target_model_filename_or_obj)
        lists = _construction.create_lsgst_circuit_lists(
            target_model, self.prep_fiducials, self.meas_fiducials, self.germs,
            self.maxlengths, self.fiducial_pairs, self.truncation_method, self.nested,
            self.fpr_keep_fraction, self.fpr_keep_seed, self.include_lgst,
            self.aliases, self.circuit_rules, dscheck, action_if_missing,
            self.germ_length_limits, verbosity)
        #FUTURE: add support for "advanced options" (probably not in __init__ though?):
        # trunc_scheme=advancedOptions.get('truncScheme', "whole germ powers")

        super().__init__(target_model, lists, None, qubit_labels, self.nested)
        self.auxfile_types['prep_fiducials'] = 'text-circuit-list'
        self.auxfile_types['meas_fiducials'] = 'text-circuit-list'
        self.auxfile_types['germs'] = 'text-circuit-list'
        self.auxfile_types['germ_length_limits'] = 'pickle'
        self.auxfile_types['fiducial_pairs'] = 'pickle'
        if add_default_protocol:
            self.add_default_protocol(StandardGST(name='StdGST'))

    def copy_with_maxlengths(self, max_lengths, germ_length_limits=None,
                             dscheck=None, action_if_missing='raise', verbosity=0):
        """
        Copies this GST experiment design to one with the same data except a different set of maximum lengths.

        Parameters
        ----------
        max_lengths_to_keep : list
            A list of the maximum lengths that should be present in the
            returned experiment design.

        germ_length_limits : dict, optional
            A dictionary limiting the max-length values to keep for specific germs.
            Keys are germ sequences and values are integers.  If `None`, then the
            current length limits are used.

        dscheck : DataSet, optional
            A data set which filters the circuits used for GST. When a standard-GST
            circuit is missing from this `DataSet`, action is taken according to
            `action_if_missing`.

        action_if_missing : {"raise","drop"}, optional
            The action to take when a desired circuit is missing from
            `dscheck` (only relevant when `dscheck` is not None).  "raise" causes
            a ValueError to be raised; "drop" causes the missing sequences to be
            dropped from the returned set.

        -------
        StandardGSTDesign
        """
        if germ_length_limits is None:
            gll = self.germ_length_limits
        else:
            gll = self.germ_length_limits.copy() if (self.germ_length_limits is not None) else {}
            gll.update(germ_length_limits)

        ret = StandardGSTDesign(self.target_model, self.prep_fiducials, self.meas_fiducials,
                                self.germs, max_lengths, gll, self.fiducial_pairs,
                                self.fpr_keep_fraction, self.fpr_keep_seed, self.include_lgst, self.nested,
                                self.circuit_rules, self.aliases, dscheck, action_if_missing, self.qubit_labels,
                                verbosity, add_default_protocol=False)

        #filter the circuit lists in `ret` using those in `self` (in case self includes only a subset of
        # the circuits dictated by the germs, fiducials, and  fidpairs).
        return ret.truncate_to_design(self)


class GSTInitialModel(object):
    """
    Specification of a starting point for GST.

    Parameters
    ----------
    model : Model, optional
        The model to start at, given explicitly.

    starting_point : {"target", "User-supplied-Model", "LGST", "LGST-if-possible"}, optional
        The starting point type.  If `None`, then defaults to `"User-supplied-Model"` if
        `model` is given, otherwise to `"target"`.

    depolarize_start : float, optional
        Amount to depolarize the starting model just prior to running GST.

    randomize_start : float, optional
        Amount to randomly kick the starting model just prior to running GST.

    lgst_gaugeopt_tol : float, optional
        Gauge-optimization tolerance for the post-LGST gauge optimization that is
        performed when `starting_point == "LGST"` or possibly when `"starting_point == "LGST-if-possible"`.

    contract_start_to_cptp : bool, optional
        Whether the Model should be forced ("contracted") to being CPTP just prior to running GST.
    """

    @classmethod
    def cast(cls, obj):
        """
        Cast `obj` to a :class:`GSTInitialModel` object.

        Parameters
        ----------
        obj : object
            object to cast.  Can be a `GSTInitialModel` (naturally) or a :class:`Model`.

        Returns
        -------
        GSTInitialModel
        """
        return obj if isinstance(obj, GSTInitialModel) else cls(obj)

    def __init__(self, model=None, starting_point=None, depolarize_start=0, randomize_start=0,
                 lgst_gaugeopt_tol=1e-6, contract_start_to_cptp=False):
        # Note: starting_point can be an initial model or string
        self.model = model
        if starting_point is None:
            self.starting_point = "target" if (model is None) else "User-supplied-Model"
        else:
            self.starting_point = starting_point

        self.lgst_gaugeopt_tol = lgst_gaugeopt_tol
        self.contract_start_to_cptp = contract_start_to_cptp
        self.depolarize_start = depolarize_start
        self.randomize_start = randomize_start

    def get_model(self, edesign, gaugeopt_target, dataset, comm):
        """
        Retrieve the starting-point :class:`Model` used to seed a long-sequence GST run.

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design containing the circuits being used, the qubit labels,
            and (possibly) a target model (for use when `starting_point == "target"`) and
            fiducial circuits (for LGST).

        gaugeopt_target : Model
            The gauge-optimization target, i.e. distance to this model is the objective function
            within the post-LGST gauge-optimization step.

        dataset : DataSet
            Data used to execute LGST when needed.

        comm : mpi4py.MPI.Comm
            A MPI communicator to divide workload amoung multiple processors.

        Returns
        -------
        Model
        """

        target_model = edesign.target_model if isinstance(edesign, HasTargetModel) else None

        #Get starting point (model), which is used to compute other quantities
        # Note: should compute on rank 0 and distribute?
        starting_pt = self.starting_point
        if starting_pt == "User-supplied-Model":
            mdl_start = self.model

        elif starting_pt in ("LGST", "LGST-if-possible"):
            #lgst_advanced = advancedOptions.copy(); lgst_advanced.update({'estimateLabel': "LGST", 'onBadFit': []})
            mdl_start = self.model if (self.model is not None) else target_model
            if mdl_start is None:
                raise ValueError(("LGST requires a model. Specify an initial model or use an experiment"
                                  " design with a target model"))

            lgst = LGST(mdl_start,
                        gaugeopt_suite={'lgst_gaugeopt': {'tol': self.lgst_gaugeopt_tol}},
                        gaugeopt_target=gaugeopt_target, badfit_options=None, name="LGST")

            try:  # see if LGST can be run on this data
                if isinstance(edesign, StandardGSTDesign) and len(edesign.maxlengths) > 0:
                    lgst_design = edesign.copy_with_maxlengths([edesign.maxlengths[0]], dscheck=dataset,
                                                               action_if_missing='drop')
                else:
                    lgst_design = edesign  # just use the whole edesign
                lgst_data = _proto.ProtocolData(lgst_design, dataset)
                lgst.check_if_runnable(lgst_data)
                starting_pt = "LGST"
            except ValueError as e:
                if starting_pt == "LGST": raise e  # error if we *can't* run LGST

                #Fall back to target or custom model
                if self.model is not None:
                    starting_pt = "User-supplied-Model"
                    mdl_start = self.model
                else:
                    starting_pt = "target"
                    mdl_start = target_model

            if starting_pt == "LGST":
                lgst_results = lgst.run(lgst_data)
                mdl_start = lgst_results.estimates['LGST'].models['lgst_gaugeopt']

        elif starting_pt == "target":
            mdl_start = target_model
        else:
            raise ValueError("Invalid starting point: %s" % starting_pt)

        if mdl_start is None:
            raise ValueError("Could not create or obtain an initial model!")

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
    """
    Options for post-processing a GST fit that was unsatisfactory.

    Parameters
    ----------
    threshold : float, optional
        A threshold, given in number-of-standard-deviations, below which a
        GST fit is considered satisfactory (and no "bad-fit" processing is needed).

    actions : tuple, optional
        Actions to take when a GST fit is unsatisfactory.

    wildcard_budget_includes_spam : bool, optional
        Include a SPAM budget within the wildcard budget used to process
        the `"wildcard"` action.
    """

    @classmethod
    def cast(cls, obj):
        """
        Cast `obj` to a :class:`GSTBadFitOptions` object.

        Parameters
        ----------
        obj : object
            Object to cast.  Can be a `GSTBadFitOptions` (naturally) or a dictionary
            of constructor arguments.

        Returns
        -------
        GSTBadFitOptions
        """
        if isinstance(obj, GSTBadFitOptions):
            return obj
        else:  # assum obj is a dict of arguments
            return cls(**obj) if obj else cls()  # allow obj to be None => defaults

    def __init__(self, threshold=DEFAULT_BAD_FIT_THRESHOLD, actions=(),
                 wildcard_budget_includes_spam=True,
                 wildcard_L1_weights=None, wildcard_primitive_op_labels=None,
                 wildcard_initial_budget=None, wildcard_methods=('neldermead',),
                 wildcard_inadmissable_action='print'):
        valid_actions = ('wildcard', 'Robust+', 'Robust', 'robust+', 'robust', 'do nothing')
        if not all([(action in valid_actions) for action in actions]):
            raise ValueError("Invalid action in %s! Allowed actions are %s" % (str(actions), str(valid_actions)))
        self.threshold = float(threshold)
        self.actions = tuple(actions)  # e.g. ("wildcard", "Robust+"); empty list => 'do nothing'
        self.wildcard_budget_includes_spam = bool(wildcard_budget_includes_spam)
        self.wildcard_L1_weights = wildcard_L1_weights
        self.wildcard_primitive_op_labels = wildcard_primitive_op_labels
        self.wildcard_initial_budget = wildcard_initial_budget
        self.wildcard_methods = wildcard_methods
        self.wildcard_inadmissable_action = wildcard_inadmissable_action  # can be 'raise' or 'print'


class GSTObjFnBuilders(object):
    """
    Holds the objective-function builders needed for long-sequence GST.

    Parameters
    ----------
    iteration_builders : list or tuple
        A list of :class:`ObjectiveFunctionBuilder` objects used (sequentially)
        on each GST iteration.

    final_builders : list or tuple, optional
        A list of :class:`ObjectiveFunctionBuilder` objects used (sequentially)
        on the final GST iteration.
    """

    @classmethod
    def cast(cls, obj):
        """
        Cast `obj` to a :class:`GSTObjFnBuilders` object.

        Parameters
        ----------
        obj : object
            Object to cast.  Can be a `GSTObjFnBuilders` (naturally), a
            dictionary of :method:`create_from` arguments (or None), or a
            list or tuple of the `(iteration_builders, final_builders)` constructor arguments.

        Returns
        -------
        GSTObjFnBuilders
        """
        if isinstance(obj, cls): return obj
        elif obj is None: return cls.create_from()
        elif isinstance(obj, dict): return cls.create_from(**obj)
        elif isinstance(obj, (list, tuple)): return cls(*obj)
        else: raise ValueError("Cannot create an %s object from '%s'" % (cls.__name__, str(type(obj))))

    @classmethod
    def create_from(cls, objective='logl', freq_weighted_chi2=False, always_perform_mle=False, only_perform_mle=False):
        """
        Creates a common :class:`GSTObjFnBuilders` object from several arguments.

        Parameters
        ----------
        objective : {'logl', 'chi2'}, optional
            Whether to create builders for maximum-likelihood or minimum-chi-squared GST.

        freq_weighted_chi2 : bool, optional
            Whether chi-squared objectives use frequency-weighting.  If you're not sure
            what this is, leave it as `False`.

        always_perform_mle : bool, optional
            Perform a ML-GST step on *each* iteration (usually this is only done for the
            final iteration).

        only_perform_mle : bool, optional
            Only perform a ML-GST step on each iteration, i.e. do *not* perform any chi2
            minimization to "seed" the ML-GST step.

        Returns
        -------
        GSTObjFnBuilders
        """
        chi2_builder = _objfns.ObjectiveFunctionBuilder.create_from('chi2', freq_weighted_chi2)
        mle_builder = _objfns.ObjectiveFunctionBuilder.create_from('logl')

        if objective == "chi2":
            iteration_builders = [chi2_builder]
            final_builders = []

        elif objective == "logl":
            if always_perform_mle:
                iteration_builders = [mle_builder] if only_perform_mle else [chi2_builder, mle_builder]
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
    """
    The core gate set tomography protocol, which optimizes a parameterized model to (best) fit a data set.

    Parameters
    ----------
    initial_model : Model or GSTInitialModel, optional
        The starting-point Model.

    gaugeopt_suite : str or list or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  A string or
        list of strings (see below) specifies built-in sets of gauge optimizations,
        otherwise `gaugeopt_suite` should be a dictionary of gauge-optimization
        parameter dictionaries (arguments to :func:`gaugeopt_to_target`).  The key
        names of `gaugeopt_suite` then label the gauge optimizations within the
        resuling `Estimate` objects.  The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        (given by the target model), which are used as the default when
        `gaugeopt_target` is None.

    objfn_builders : GSTObjFnBuilders, optional
        The objective function(s) to optimize.  Can also be anything that can
        be cast to a :class:`GSTObjFnBuilders` object.

    optimizer : Optimizer, optional
        The optimizer to use.  Can also be anything that can be case to a :class:`Optimizer`.

    badfit_options : GSTBadFitOptions, optional
        Options specifying what post-processing actions should be performed if the GST
        fit is unsatisfactory.  Can also be anything that can be cast to a
        :class:`GSTBadFitOptions` object.

    verbosity : int, optional
        The 'verbosity' option is an integer specifying the level of
        detail printed to stdout during the calculation.

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    def __init__(self, initial_model=None, gaugeopt_suite='stdgaugeopt',
                 gaugeopt_target=None, objfn_builders=None, optimizer=None,
                 badfit_options=None, verbosity=2, name=None):
        super().__init__(name)
        self.initial_model = GSTInitialModel.cast(initial_model)
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.badfit_options = GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        if isinstance(optimizer, _opt.Optimizer):
            self.optimizer = optimizer
        else:
            if optimizer is None: optimizer = {}
            if 'first_fditer' not in optimizer:  # then add default first_fditer value
                mdl = self.initial_model.model
                optimizer['first_fditer'] = 0 if mdl and isinstance(mdl.sim, _TermFSim) else 1
            self.optimizer = _opt.CustomLMOptimizer.cast(optimizer)

        objfn_builders = GSTObjFnBuilders.cast(objfn_builders)
        self.iteration_builders = objfn_builders.iteration_builders
        self.final_builders = objfn_builders.final_builders

        self.auxfile_types['initial_model'] = 'pickle'
        self.auxfile_types['badfit_options'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['optimizer'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['iteration_builders'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['final_builders'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['gaugeopt_suite'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['gaugeopt_target'] = 'pickle'  # TODO - better later? - json?

        #Advanced options that could be changed by users who know what they're doing
        #self.estimate_label = estimate_label -- just use name?
        self.profile = 1
        self.record_output = True
        self.distribute_method = "default"
        self.oplabel_aliases = None
        self.circuit_weights = None
        self.unreliable_ops = ('Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz')

    #TODO: Maybe make methods like this separate functions??
    #def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, max_lengths):
    #    design = StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, max_lengths)
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
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        ModelEstimateResults
        """
        tref = _time.time()

        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _objs.VerbosityPrinter.create_printer(self.verbosity, comm)
        if self.record_output and not printer.is_recording():
            printer.start_recording()

        resource_alloc = _ResourceAllocation(comm, memlimit, profiler,
                                             distribute_method=self.distribute_method)

        circuit_lists = data.edesign.circuit_lists
        aliases = circuit_lists[-1].op_label_aliases if isinstance(circuit_lists[-1], _CircuitList) else None
        ds = data.dataset

        if self.oplabel_aliases:  # override any other aliases with ones specifically given
            aliases = self.oplabel_aliases

        bulk_circuit_lists = [_CircuitList(lst, aliases, self.circuit_weights)
                              for lst in circuit_lists]

        tnxt = _time.time(); profiler.add_time('GST: loading', tref); tref = tnxt
        mdl_start = self.initial_model.get_model(data.edesign, self.gaugeopt_target, data.dataset, comm)

        tnxt = _time.time(); profiler.add_time('GST: Prep Initial seed', tref); tref = tnxt

        #Run Long-sequence GST on data
        mdl_lsgst_list, optimums_list, final_store = _alg.run_iterative_gst(
            ds, mdl_start, bulk_circuit_lists, self.optimizer,
            self.iteration_builders, self.final_builders,
            resource_alloc, printer)

        tnxt = _time.time(); profiler.add_time('GST: total iterative optimization', tref); tref = tnxt

        #set parameters
        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        parameters['final_objfn_builder'] = self.final_builders[-1] if len(self.final_builders) > 0 \
            else self.iteration_builders[-1]
        parameters['final_objfn_store'] = final_store  # Final obj. function evaluated at best-fit point (cache too)
        parameters['profiler'] = profiler
        # Note: we associate 'final_cache' with the Estimate, which means we assume that *all*
        # of the models in the estimate can use same evaltree, have the same default prep/POVMs, etc.

        #TODO: add qtys abot fit from optimums_list

        # TODO: use final_store more fully - it may contain useful cached quantities for creating
        # the estimate and gaugeopt/badfit below
        ret = ModelEstimateResults(data, self)
        estimate = _Estimate.create_gst_estimate(ret, data.edesign.target_model, mdl_start, mdl_lsgst_list, parameters)
        ret.add_estimate(estimate, estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, final_store, data.edesign.target_model,
                                        self.gaugeopt_suite, self.gaugeopt_target, self.unreliable_ops,
                                        self.badfit_options, parameters['final_objfn_builder'], self.optimizer,
                                        resource_alloc, printer)


class LinearGateSetTomography(_proto.Protocol):
    """
    The linear gate set tomography protocol.

    Parameters
    ----------
    target_model : Model, optional
        The target (ideal) model.

    gaugeopt_suite : str or list or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  A string or
        list of strings (see below) specifies built-in sets of gauge optimizations,
        otherwise `gaugeopt_suite` should be a dictionary of gauge-optimization
        parameter dictionaries (arguments to :func:`gaugeopt_to_target`).  The key
        names of `gaugeopt_suite` then label the gauge optimizations within the
        resuling `Estimate` objects.  The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by `target_model`, which are used as the default when
        `gaugeopt_target` is None.

    badfit_options : GSTBadFitOptions, optional
        Options specifying what post-processing actions should be performed if the LGST
        fit is unsatisfactory.  Can also be anything that can be cast to a
        :class:`GSTBadFitOptions` object.

    verbosity : int, optional
        The 'verbosity' option is an integer specifying the level of
        detail printed to stdout during the calculation.

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    def __init__(self, target_model=None, gaugeopt_suite='stdgaugeopt', gaugeopt_target=None,
                 badfit_options=None, verbosity=2, name=None):
        super().__init__(name)
        self.target_model = target_model
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.badfit_options = GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        #Advanced options that could be changed by users who know what they're doing
        self.profile = 1
        self.record_output = True
        self.oplabels = "default"
        self.oplabel_aliases = None
        self.unreliable_ops = ('Gcnot', 'Gcphase', 'Gms', 'Gcn', 'Gcx', 'Gcz')

    def check_if_runnable(self, data):
        """
        Raises a ValueError if LGST cannot be run on data

        Parameters
        ----------
        data : ProtocolData
            The experimental data to test for LGST-compatibility.

        Returns
        -------
        None
        """
        edesign = data.edesign

        if not isinstance(edesign, StandardGSTDesign):
            raise ValueError("LGST must be given a `StandardGSTDesign` experiment design (for fiducial circuits)!")

        if len(edesign.circuit_lists) != 1:
            raise ValueError("There must be at exactly one circuit list in the input experiment design!")

        target_model = self.target_model if (self.target_model is not None) else edesign.target_model
        if isinstance(target_model, _objs.ExplicitOpModel):
            if not all([(isinstance(g, _objs.FullDenseOp)
                         or isinstance(g, _objs.TPDenseOp))
                        for g in target_model.operations.values()]):
                raise ValueError("LGST can only be applied to explicit models with dense operators")
        else:
            raise ValueError("LGST can only be applied to explicit models with dense operators")

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        ModelEstimateResults
        """
        self.check_if_runnable(data)

        edesign = data.edesign
        target_model = self.target_model if (self.target_model is not None) else edesign.target_model

        if isinstance(edesign, _proto.CircuitListsDesign):
            circuit_list = edesign.circuit_lists[0]
        else:
            circuit_list = edesign.all_circuits_needing_data  # Never reached, since design must be a StandardGSTDesign!
        circuit_list = _CircuitList.cast(circuit_list)

        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _objs.Profiler(comm, False)
        elif profile == 2: profiler = _objs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _objs.VerbosityPrinter.create_printer(self.verbosity, comm)
        if self.record_output and not printer.is_recording():
            printer.start_recording()

        resource_alloc = _ResourceAllocation(comm, memlimit, profiler,
                                             distribute_method="default")

        ds = data.dataset

        aliases = circuit_list.op_label_aliases if self.oplabel_aliases is None else self.oplabel_aliases
        op_labels = self.oplabels if self.oplabels != "default" else \
            list(target_model.operations.keys()) + list(target_model.instruments.keys())

        # Note: this returns a model with the *same* parameterizations as target_model
        mdl_lgst = _alg.run_lgst(ds, edesign.prep_fiducials, edesign.meas_fiducials, target_model,
                                 op_labels, svd_truncate_to=target_model.dim,
                                 op_label_aliases=aliases, verbosity=printer)
        final_store = _objs.ModelDatasetCircuitsStore(mdl_lgst, ds, circuit_list, resource_alloc,
                                                      array_types=('E',), verbosity=printer)

        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        parameters['profiler'] = profiler
        parameters['final_objfn_store'] = final_store
        parameters['final_objfn_builder'] = _objfns.PoissonPicDeltaLogLFunction.builder()
        # just set final objective function as default logl objective (for ease of later comparison)

        ret = ModelEstimateResults(data, self)
        estimate = _Estimate(ret, {'target': target_model, 'seed': target_model, 'lgst': mdl_lgst,
                                   'iteration estimates': [mdl_lgst],
                                   'final iteration estimate': mdl_lgst},
                             parameters)
        ret.add_estimate(estimate, estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, final_store, data.edesign.target_model, self.gaugeopt_suite,
                                        self.gaugeopt_target, self.unreliable_ops, self.badfit_options,
                                        None, None, resource_alloc, printer)


#HERE's what we need to do:
#x continue upgrading this module: StandardGST (similar to others, but need "appendTo" workaround)
#x upgrade ModelTest
#x fix do_XXX driver functions in longsequence.py
#x (maybe upgraded advancedOptions there to a class w/validation, etc)
#x upgrade likelihoodfns.py and chi2.py to use objective funtions -- should be lots of consolidation, and maybe
#x add hessian & non-poisson-pic logl to objective fns.
# fix report generation (changes to ModelEstimateResults)
# run/update tests - test out custom/new objective functions.
class StandardGST(_proto.Protocol):
    """
    The standard-practice GST protocol.

    Parameters
    ----------
    modes : str, optional
        A comma-separated list of modes which dictate what types of analyses
        are performed.  Currently, these correspond to different types of
        parameterizations/constraints to apply to the estimated model.
        The default value is usually fine.  Allowed values are:

        - "full" : full (completely unconstrained)
        - "TP"   : TP-constrained
        - "CPTP" : Lindbladian CPTP-constrained
        - "H+S"  : Only Hamiltonian + Stochastic errors allowed (CPTP)
        - "S"    : Only Stochastic errors allowed (CPTP)
        - "Target" : use the target (ideal) gates as the estimate
        - <model> : any key in the `models_to_test` argument

    gaugeopt_suite : str or list or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  A string or
        list of strings (see below) specifies built-in sets of gauge optimizations,
        otherwise `gaugeopt_suite` should be a dictionary of gauge-optimization
        parameter dictionaries (arguments to :func:`gaugeopt_to_target`).  The key
        names of `gaugeopt_suite` then label the gauge optimizations within the
        resuling `Estimate` objects.  The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by the target model, which are used as the default when
        `gaugeopt_target` is None.

    models_to_test : dict, optional
        A dictionary of Model objects representing (gate-set) models to
        test against the data.  These Models are essentially hypotheses for
        which (if any) model generated the data.  The keys of this dictionary
        can (and must, to actually test the models) be used within the comma-
        separate list given by the `modes` argument.

    objfn_builders : GSTObjFnBuilders, optional
        The objective function(s) to optimize.  Can also be anything that can
        be cast to a :class:`GSTObjFnBuilders` object.  Applies to all modes.

    optimizer : Optimizer, optional
        The optimizer to use.  Can also be anything that can be case to a
        :class:`Optimizer`.  Applies to all modes.

    badfit_options : GSTBadFitOptions, optional
        Options specifying what post-processing actions should be performed if the GST
        fit is unsatisfactory.  Can also be anything that can be cast to a
        :class:`GSTBadFitOptions` object.  Applies to all modes.

    verbosity : int, optional
        The 'verbosity' option is an integer specifying the level of
        detail printed to stdout during the calculation.

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    def __init__(self, modes="TP,CPTP,Target",
                 gaugeopt_suite='stdgaugeopt',
                 gaugeopt_target=None, models_to_test=None,
                 objfn_builders=None, optimizer=None,
                 badfit_options=None, verbosity=2, name=None):

        super().__init__(name)
        self.modes = modes.split(',')
        self.models_to_test = models_to_test
        self.gaugeopt_suite = gaugeopt_suite
        self.gaugeopt_target = gaugeopt_target
        self.objfn_builders = objfn_builders
        self.optimizer = optimizer
        self.badfit_options = GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        self.auxfile_types['models_to_test'] = 'pickle'
        self.auxfile_types['gaugeopt_suite'] = 'pickle'
        self.auxfile_types['gaugeopt_target'] = 'pickle'
        self.auxfile_types['objfn_builders'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['optimizer'] = 'pickle'  # TODO - better later? - json?
        self.auxfile_types['badfit_options'] = 'pickle'  # TODO - better later? - json?

        #Advanced options that could be changed by users who know what they're doing
        self.starting_point = {}  # a dict whose keys are modes

    #def run_using_germs_and_fiducials(self, dataset, target_model, prep_fiducials, meas_fiducials, germs, max_lengths):
    #    design = StandardGSTDesign(target_model, prep_fiducials, meas_fiducials, germs, max_lengths)
    #    data = _proto.ProtocolData(design, dataset)
    #    return self.run(data)

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        ProtocolResults
        """
        printer = _objs.VerbosityPrinter.create_printer(self.verbosity, comm)

        modes = self.modes
        models_to_test = self.models_to_test
        if models_to_test is None: models_to_test = {}

        # Choose an objective function to use for model testing
        mt_builder = None  # None => use the default builder
        if self.objfn_builders is not None:
            if len(self.objfn_builders.final_builders) > 0:
                mt_builder = self.objfn_builders.final_builders[0]
            elif len(self.objfn_builders.iteration_builders) > 0:
                mt_builder = self.objfn_builders.iteration_builders[0]

        ret = ModelEstimateResults(data, self)
        with printer.progress_logging(1):
            for i, mode in enumerate(modes):
                printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

                if mode == "Target":
                    model_to_test = data.edesign.target_model.copy()  # no parameterization change
                    mdltest = _ModelTest(model_to_test, None, self.gaugeopt_suite, self.gaugeopt_target,
                                         mt_builder, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)

                elif mode in models_to_test:
                    mdltest = _ModelTest(models_to_test[mode], None, self.gaugeopt_suite, self.gaugeopt_target,
                                         None, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)

                else:
                    #Try to interpret `mode` as a parameterization
                    parameterization = mode  # for now, 1-1 correspondence
                    initial_model = data.edesign.target_model.copy()

                    try:
                        initial_model.set_all_parameterizations(parameterization)
                    except ValueError as e:
                        raise ValueError("Could not interpret '%s' mode as a parameterization! Details:\n%s"
                                         % (mode, str(e)))

                    initial_model = GSTInitialModel(initial_model, self.starting_point.get(mode, None))
                    gst = GST(initial_model, self.gaugeopt_suite, self.gaugeopt_target, self.objfn_builders,
                              self.optimizer, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = gst.run(data, memlimit, comm)
                    ret.add_estimates(result)

        return ret


# ------------------ HELPER FUNCTIONS -----------------------------------

def gaugeopt_suite_to_dictionary(gaugeopt_suite, model, unreliable_ops=(), verbosity=0):
    """
    Creates gauge-optimization dictionaries from "suite" names.

    Constructs a dictionary of gauge-optimization parameter dictionaries based
    on "gauge optimization suite" name(s).

    This is primarily a helper function for :func:`run_stdpractice_gst`, but can
    be useful in its own right for constructing the would-be gauge optimization
    dictionary used in :func:`run_stdpractice_gst` and modifying it slightly before
    before passing it in (`run_stdpractice_gst` will accept a raw dictionary too).

    Parameters
    ----------
    gaugeopt_suite : str or dict, optional
        Specifies which gauge optimizations to perform on each estimate.  An string
        (see below) specifies a built-in set of gauge optimizations, otherwise
        `gaugeopt_suite` should be a dictionary of gauge-optimization parameter
        dictionaries (arguments to :func:`gaugeopt_to_target`).  The key names of
        `gaugeopt_suite` then label the gauge optimizations within the resuling
        `Estimate` objects.  The built-in gauge optmization suites are:

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

    unreliable_ops : tuple, optional
        A tuple of gate (or circuit-layer) labels that count as "unreliable operations".
        Typically these are the multi-qubit (2-qubit) gates.

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
    printer = _objs.VerbosityPrinter.create_printer(verbosity)

    if gaugeopt_suite is None:
        gaugeopt_suite = {}
    elif isinstance(gaugeopt_suite, str):
        gaugeopt_suite = {gaugeopt_suite: gaugeopt_suite}
    elif isinstance(gaugeopt_suite, tuple):
        gaugeopt_suite = {nm: nm for nm in gaugeopt_suite}

    assert(isinstance(gaugeopt_suite, dict)), \
        "Can't convert type '%s' to a gauge optimization suite dictionary!" % str(type(gaugeopt_suite))

    #Build ordered dict of gauge optimization parameters
    gaugeopt_suite_dict = _collections.OrderedDict()
    for lbl, goparams in gaugeopt_suite.items():
        if isinstance(goparams, str):
            _update_gaugeopt_dict_from_suitename(gaugeopt_suite_dict, lbl, goparams,
                                                 model, unreliable_ops, printer)
        elif hasattr(goparams, 'keys'):
            gaugeopt_suite_dict[lbl] = goparams.copy()
            gaugeopt_suite_dict[lbl].update({'verbosity': printer})
        else:
            assert(isinstance(goparams, list)), "If not a dictionary, gauge opt params should be a list of dicts!"
            gaugeopt_suite_dict[lbl] = []
            for goparams_stage in goparams:
                dct = goparams_stage.copy()
                dct.update({'verbosity': printer})
                gaugeopt_suite_dict[lbl].append(dct)

    return gaugeopt_suite_dict


def _update_gaugeopt_dict_from_suitename(gaugeopt_suite_dict, root_lbl, suite_name, model, unreliable_ops, printer):
    if suite_name in ("stdgaugeopt", "stdgaugeopt-unreliable2Q", "stdgaugeopt-tt"):

        stages = []  # multi-stage gauge opt
        gg = model.default_gauge_group
        if isinstance(gg, _objs.TrivialGaugeGroup):
            if suite_name == "stdgaugeopt-unreliable2Q" and model.dim == 16:
                if any([gl in model.operations.keys() for gl in unreliable_ops]):
                    gaugeopt_suite_dict[root_lbl] = {'verbosity': printer}
            else:
                #just do a single-stage "trivial" gauge opts using default group
                gaugeopt_suite_dict[root_lbl] = {'verbosity': printer}

        elif gg is not None:
            metric = 'frobeniustt' if suite_name == 'stdgaugeopt-tt' else 'frobenius'

            #Stage 1: plain vanilla gauge opt to get into "right ballpark"
            if gg.name in ("Full", "TP"):
                stages.append(
                    {
                        'gates_metric': metric, 'spam_metric': metric,
                        'item_weights': {'gates': 1.0, 'spam': 1.0},
                        'verbosity': printer
                    })

            #Stage 2: unitary gauge opt that tries to nail down gates (at
            #         expense of spam if needed)
            stages.append(
                {
                    'gates_metric': metric, 'spam_metric': metric,
                    'item_weights': {'gates': 1.0, 'spam': 0.0},
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
                    'gates_metric': metric, 'spam_metric': metric,
                    'item_weights': {'gates': 0.0, 'spam': 1.0},
                    'spam_penalty_factor': 1.0,
                    'gauge_group': s3gg(model.dim),
                    'oob_check_interval': 1,
                    'verbosity': printer
                })

            if suite_name == "stdgaugeopt-unreliable2Q" and model.dim == 16:
                if any([gl in model.operations.keys() for gl in unreliable_ops]):
                    stage2_item_weights = {'gates': 1, 'spam': 0.0}
                    for gl in unreliable_ops:
                        if gl in model.operations.keys(): stage2_item_weights[gl] = 0.01
                    stages_2qubit_unreliable = [stage.copy() for stage in stages]  # ~deep copy of stages
                    istage2 = 1 if gg.name in ("Full", "TP") else 0
                    stages_2qubit_unreliable[istage2]['item_weights'] = stage2_item_weights
                    gaugeopt_suite_dict[root_lbl] = stages_2qubit_unreliable  # add additional gauge opt
                else:
                    _warnings.warn(("`unreliable2Q` was given as a gauge opt suite, but none of the"
                                    " gate names in 'unreliable_ops', i.e., %s,"
                                    " are present in the target model.  Omitting 'single-2QUR' gauge opt.")
                                   % (", ".join(unreliable_ops)))
            else:
                gaugeopt_suite_dict[root_lbl] = stages  # can be a list of stage dictionaries

    elif suite_name in ("varySpam", "varySpamWt", "varyValidSpamWt", "toggleValidSpam") or \
        suite_name in ("varySpam-unreliable2Q", "varySpamWt-unreliable2Q",
                       "varyValidSpamWt-unreliable2Q", "toggleValidSpam-unreliable2Q"):

        base_wts = {'gates': 1}
        if suite_name.endswith("unreliable2Q") and model.dim == 16:
            if any([gl in model.operations.keys() for gl in unreliable_ops]):
                base = {'gates': 1}
                for gl in unreliable_ops:
                    if gl in model.operations.keys(): base[gl] = 0.01
                base_wts = base

        if suite_name == "varySpam":
            valid_spam_range = [0, 1]; spamwt_range = [1e-4, 1e-1]
        elif suite_name == "varySpamWt":
            valid_spam_range = [0]; spamwt_range = [1e-4, 1e-1]
        elif suite_name == "varyValidSpamWt":
            valid_spam_range = [1]; spamwt_range = [1e-4, 1e-1]
        elif suite_name == "toggleValidSpam":
            valid_spam_range = [0, 1]; spamwt_range = [1e-3]
        else:
            valid_spam_range = []
            spamwt_range = []

        if suite_name == root_lbl:  # then shorten the root name
            root_lbl = "2QUR-" if suite_name.endswith("unreliable2Q") else ""

        for valid_spam in valid_spam_range:
            for spam_weight in spamwt_range:
                lbl = root_lbl + "Spam %g%s" % (spam_weight, "+v" if valid_spam else "")
                item_weights = base_wts.copy()
                item_weights['spam'] = spam_weight
                gaugeopt_suite_dict[lbl] = {
                    'item_weights': item_weights,
                    'spam_penalty_factor': valid_spam, 'verbosity': printer}

    elif suite_name == "unreliable2Q":
        raise ValueError(("unreliable2Q is no longer a separate 'suite'.  You should precede it with the suite name, "
                          "e.g. 'stdgaugeopt-unreliable2Q' or 'varySpam-unreliable2Q'"))
    elif suite_name == "none":
        pass  # add nothing
    else:
        raise ValueError("Unknown gauge-optimization suite '%s'" % suite_name)


def _load_model(model_filename_or_obj):
    if isinstance(model_filename_or_obj, str):
        return _io.load_model(model_filename_or_obj)
    else:
        return model_filename_or_obj  # assume a Model object


def _load_fiducials_and_germs(prep_fiducial_list_or_filename,
                              meas_fiducial_list_or_filename,
                              germ_list_or_filename):

    if isinstance(prep_fiducial_list_or_filename, str):
        prep_fiducials = _io.load_circuit_list(prep_fiducial_list_or_filename)
    else: prep_fiducials = prep_fiducial_list_or_filename

    if meas_fiducial_list_or_filename is None:
        meas_fiducials = prep_fiducials  # use same strings for meas_fiducials if meas_fiducial_list_or_filename is None
    else:
        if isinstance(meas_fiducial_list_or_filename, str):
            meas_fiducials = _io.load_circuit_list(meas_fiducial_list_or_filename)
        else: meas_fiducials = meas_fiducial_list_or_filename

    #Get/load germs
    if isinstance(germ_list_or_filename, str):
        germs = _io.load_circuit_list(germ_list_or_filename)
    else: germs = germ_list_or_filename

    return prep_fiducials, meas_fiducials, germs


def _load_dataset(data_filename_or_set, comm, verbosity):
    """Loads a DataSet from the data_filename_or_set argument of functions in this module."""
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)
    if isinstance(data_filename_or_set, str):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(data_filename_or_set)[1] == ".pkl":
                with open(data_filename_or_set, 'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.load_dataset(data_filename_or_set, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = data_filename_or_set  # assume a Dataset object

    return ds


def _add_gaugeopt_and_badfit(results, estlbl, mdc_store, target_model, gaugeopt_suite, gaugeopt_target,
                             unreliable_ops, badfit_options, objfn_builder, optimizer, resource_alloc, printer):
    tref = _time.time()
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler
    model_to_gaugeopt = mdc_store.model

    #Do final gauge optimization to *final* iteration result only
    if gaugeopt_suite:
        gaugeopt_target = gaugeopt_target if gaugeopt_target else target_model
        _add_gauge_opt(results, estlbl, gaugeopt_suite, gaugeopt_target,
                       model_to_gaugeopt, unreliable_ops, comm, printer - 1)
    profiler.add_time('%s: gauge optimization' % estlbl, tref); tref = _time.time()

    _add_badfit_estimates(results, estlbl, badfit_options, objfn_builder, optimizer, resource_alloc, printer)
    profiler.add_time('%s: add badfit estimates' % estlbl, tref); tref = _time.time()

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        results.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    return results


##TODO REMOVE
#def OLD_package_into_results(callerProtocol, data, target_model, mdl_start, lsgstLists,
#                          parameters, mdl_lsgst_list, gaugeopt_suite, gaugeopt_target,
#                          comm, memLimit, output_pkl, verbosity,
#                          profiler, evaltree_cache=None):
#    # advanced_options, opt_args,
#    """
#    Performs all of the post-optimization processing common to
#    run_long_sequence_gst and do_model_evaluation.
#
#    Creates a Results object to be returned from run_long_sequence_gst
#    and do_model_evaluation (passed in as 'callerName').  Performs
#    gauge optimization, and robust data scaling (with re-optimization
#    if needed and opt_args is not None - i.e. only for
#    run_long_sequence_gst).
#    """
#    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)
#    tref = _time.time()
#    callerName = callerProtocol.name
#
#    #ret = advancedOptions.get('appendTo', None)
#    #if ret is None:
#    ret = ModelEstimateResults(data, callerProtocol)
#    #else:
#    #    # a dummy object to check compatibility w/ret2
#    #    dummy = ModelEstimateResults(data, callerProtocol)
#    #    ret.add_estimates(dummy)  # does nothing, but will complain when appropriate
#
#    #add estimate to Results
#    profiler.add_time('%s: results initialization' % callerName, tref); tref = _time.time()
#
#    #Do final gauge optimization to *final* iteration result only
#    if gaugeopt_suite:
#        if gaugeopt_target is None: gaugeopt_target = target_model
#        _add_gauge_opt(ret, estlbl, gaugeopt_suite, gaugeopt_target,
#                      mdl_lsgst_list[-1], comm, advancedOptions, printer - 1)
#        profiler.add_time('%s: gauge optimization' % callerName, tref)
#
#    #Perform extra analysis if a bad fit was obtained - do this *after* gauge-opt b/c it mimics gaugeopts
#    badFitThreshold = advancedOptions.get('badFitThreshold', DEFAULT_BAD_FIT_THRESHOLD)
#    onBadFit = advancedOptions.get('onBadFit', [])  # ["wildcard"]) #["Robust+"]) # empty list => 'do nothing'
#    badfit_opts = advancedOptions.get('badFitOptions', {'wildcard_budget_includes_spam': True,
#                                                        'wildcard_smart_init': True})
#    _add_badfit_estimates(ret, estlbl, onBadFit, badFitThreshold, badfit_opts, opt_args, evaltree_cache,
#                         comm, memLimit, printer)
#    profiler.add_time('%s: add badfit estimates' % callerName, tref); tref = _time.time()
#
#    #Add recorded info (even robust-related info) to the *base*
#    #   estimate label's "stdout" meta information
#    if printer.is_recording():
#        ret.estimates[estlbl].meta['stdout'] = printer.stop_recording()
#
#    #Write results to a pickle file if desired
#    if output_pkl and (comm is None or comm.Get_rank() == 0):
#        if isinstance(output_pkl, str):
#            with open(output_pkl, 'wb') as pklfile:
#                _pickle.dump(ret, pklfile)
#        else:
#            _pickle.dump(ret, output_pkl)
#
#    return ret


#def _add_gauge_opt(estimate, gaugeOptParams, target_model, starting_model,
#                  comm=None, verbosity=0):

def _add_gauge_opt(results, base_est_label, gaugeopt_suite, target_model, starting_model,
                   unreliable_ops, comm=None, verbosity=0):
    """
    Add a gauge optimization to an estimate.

    Parameters
    ----------
    results : ModelEstimateResults
        The parent results of the estimate to add a gauge optimization to.  The estimate is
        specified via `results` and `base_est_label` rather than just passing an :class:`Estimate`
        directly so that related (e.g. bad-fit) estimates can also be updated.

    base_est_label : str
        The key within `results.estimates` of the *primary* :class:`Estimate` to update.

    gaugeopt_suite : str or list or dict, optional
        The gauge optimization suite specifying what gauge-optimizations to perform.

    target_model : Model
        The target model, which specifies the ideal gates and the default gauge group
        to optimize over (this should be set prior to calling this function).

    starting_model : Model
        The starting model of the GST or GST-like protocol.  This communicates the
        parameterization that is available to gauge optimize over, and helps interpret
        gauge-optimization-suite names (e.g. "stdgaugeopt" produces different steps
        based on the parameterization of `starting_model`).

    unreliable_ops : tuple, optional
        A tuple of gate (or circuit-layer) labels that count as "unreliable operations".
        Typically these are the multi-qubit (2-qubit) gates.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to run computations in parallel.

    verbosity : int, optional
        The level of detail printed to stdout.

    Returns
    -------
    None
    """
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)

    #Get gauge optimization dictionary
    gaugeopt_suite_dict = gaugeopt_suite_to_dictionary(gaugeopt_suite, starting_model,
                                                       unreliable_ops, printer - 1)

    if target_model is not None:
        assert(isinstance(target_model, _objs.Model)), "`gaugeOptTarget` must be None or a Model"
        for goparams in gaugeopt_suite_dict.values():
            goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
            for goparams_dict in goparams_list:
                if 'target_model' in goparams_dict:
                    _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                    " user-defined target_model in gauge opt"
                                    " param dict(s)"))
                goparams_dict.update({'target_model': target_model})

    #Gauge optimize to list of gauge optimization parameters
    for go_label, goparams in gaugeopt_suite_dict.items():

        printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (go_label, base_est_label), 2)

        #Get starting model
        results.estimates[base_est_label].add_gaugeoptimized(goparams, None, go_label, comm, printer - 3)
        mdl_start = results.estimates[base_est_label].get_start_model(goparams)

        #Gauge optimize data-scaled estimate also
        for suffix in ROBUST_SUFFIX_LIST:
            robust_est_label = base_est_label + suffix
            if robust_est_label in results.estimates:
                mdl_start_robust = results.estimates[robust_est_label].get_start_model(goparams)

                if mdl_start_robust.frobeniusdist(mdl_start) < 1e-8:
                    printer.log("-- Conveying '%s' gauge optimization from %s to %s estimate --" %
                                (go_label, base_est_label, robust_est_label), 2)
                    params = results.estimates[base_est_label].goparameters[go_label]  # no need to copy here
                    gsopt = results.estimates[base_est_label].models[go_label].copy()
                    results.estimates[robust_est_label].add_gaugeoptimized(params, gsopt, go_label, comm, printer - 3)
                else:
                    printer.log("-- Performing '%s' gauge optimization on %s estimate --" %
                                (go_label, robust_est_label), 2)
                    results.estimates[robust_est_label].add_gaugeoptimized(goparams, None, go_label, comm, printer - 3)


def _add_badfit_estimates(results, base_estimate_label, badfit_options,
                          objfn_builder=None, optimizer=None, resource_alloc=None, verbosity=0):
    """
    Add any and all "bad fit" estimates to `results`.

    Parameters
    ----------
    results : ModelEstimateResults
        The results to add bad-fit estimates to.

    base_estimate_label : str
        The *primary* estimate label to base bad-fit additions off of.

    badfit_options : GSTBadFitOptions
        The options specifing what constitutes a "bad fit" and what actions
        to take when one occurs.

    objfn_builder : ObjectiveFunctionBuilder
        The objective function (builder) that represents the final stage of
        optimization, such that if an estimate needs to be re-optimized this is
        the objective function to minimize.

    optimizer : Optimizer
        The optimizer to perform re-optimization, if any is needed.

    resource_alloc : ResourceAllocation, optional
        What resources are available and how they should be distributed.

    verbosity : int, optional
        Level of detail printed to stdout.

    Returns
    -------
    None
    """

    if badfit_options is None:
        return  # nothing to do

    badfit_options = GSTBadFitOptions.cast(badfit_options)
    base_estimate = results.estimates[base_estimate_label]
    parameters = base_estimate.parameters
    mdc_store = parameters.get('final_objfn_store', None)
    if mdc_store is not None:
        circuit_list = mdc_store.circuits
        mdl = mdc_store.model
        ds = mdc_store.dataset
        resource_alloc = mdc_store.resource_alloc  # Prefer resource_alloc within mdc_store?
    else:
        circuit_list = results.circuit_lists['final']
        mdl = base_estimate.models['final iteration estimate']
        ds = results.dataset
        mdc_store = _ModelDatasetCircuitStore(mdl, ds, circuit_list, resource_alloc)

    comm = resource_alloc.comm if resource_alloc else None
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)

    if badfit_options.threshold is not None and \
       base_estimate.misfit_sigma(comm=comm) <= badfit_options.threshold:
        return  # fit is good enough - no need to add any estimates

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
            new_params['weights'] = _compute_robust_scaling(badfit_typ, mdc_store, parameters)
            if badfit_typ in ("Robust", "Robust+") and (optimizer is not None) and (objfn_builder is not None):
                mdl_reopt = _reoptimize_with_weights(mdc_store, new_params['weights'],
                                                     objfn_builder, optimizer, printer - 1)
                new_final_model = mdl_reopt

        elif badfit_typ == "wildcard":
            try:
                budget_dict = _compute_wildcard_budget(mdc_store, parameters, badfit_options, printer - 1)
                for chain_name, (unmodeled, active_constraint_list) in budget_dict.items():
                    base_estimate.parameters[chain_name + "_unmodeled_error"] = unmodeled
                    base_estimate.parameters[chain_name + "_unmodeled_active_constraints"] \
                        = active_constraint_list
                if len(budget_dict) > 0:  # also store first chain info w/empty chain name (convenience)
                    first_chain = next(iter(budget_dict))
                    unmodeled, active_constraint_list = budget_dict[first_chain]
                    base_estimate.parameters["unmodeled_error"] = unmodeled
                    base_estimate.parameters["unmodeled_active_constraints"] = active_constraint_list
            except NotImplementedError as e:
                printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                new_params['unmodeled_error'] = None
            #except AssertionError as e:
            #    printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
            #    new_params['unmodeled_error'] = None
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

        results.add_estimate(_Estimate.create_gst_estimate(results, target_model, mdl_start,
                                                           models_by_iter, new_params),
                             base_estimate_label + "." + badfit_typ)

        #Add gauge optimizations to the new estimate
        for gokey, gauge_opt_params in base_estimate.goparameters.items():
            if new_final_model is not None:
                unreliable_ops = ()  # pass this in?
                _add_gauge_opt(results, base_estimate_label + '.' + badfit_typ, {gokey: gauge_opt_params},
                               target_model, new_final_model, unreliable_ops, comm, printer - 1)
            else:
                # add same gauge-optimized result as above
                go_gs_final = base_estimate.models[gokey]
                results.estimates[base_estimate_label + '.' + badfit_typ].add_gaugeoptimized(
                    gauge_opt_params.copy(), go_gs_final, gokey, comm, printer - 1)


def _get_fit_qty(mdc_store, parameters):
    # Get by-sequence goodness of fit
    objfn_builder = parameters.get('final_objfn_builder', _objfns.PoissonPicDeltaLogLFunction.builder())
    objfn = objfn_builder.build_from_store(mdc_store)
    fitqty = objfn.chi2k_distributed_qty(objfn.percircuit())
    return fitqty


def _compute_robust_scaling(scale_typ, mdc_store, parameters):
    """
    Get the per-circuit data scaling ("weights") for a given type of robust-data-scaling.
    TODO: update docstring

    Parameters
    ----------
    scale_typ : {'robust', 'robust+', 'Robust', 'Robust+'}
        The type of robust scaling.  Captial vs. lowercase "R" doesn't
        matter to this function (it indicates whether a post-scaling
        re-optimization is performed elsewhere).  The "+" postfix distinguishes
        a "version 1" scaling (no "+"), where we drastically scale down weights
        of especially bad sequences, from a "version 2" scaling ("+"), where
        we additionaly rescale all the circuit data to achieve the desired chi2
        distribution of per-circuit goodness-of-fit values *without reordering*
        these values.

    model : Model
        The final model fit.

    ds : DataSet
        The data set to compare to the model predictions.

    circuit_list : list
        A list of the :class:`Circuit`s whose data should be compared.

    parameters : dict
        Various parameters of the estimate at hand.

    comm : mpi4py.MPI.Comm, optional
        An MPI communicator used to run this computation in parallel.

    mem_limit : int, optional
        A rough per-processor memory limit in bytes.

    Returns
    -------
    dict
        A dictionary of circuit weights.  Keys are cirrcuits and values are
        scaling factors that should be applied to the data counts for that circuit.
        Omitted circuits should not be scaled.
    """
    circuit_list = mdc_store.circuits
    ds = mdc_store.dataset

    fitqty = _get_fit_qty(mdc_store, parameters)
    #Note: fitqty[iCircuit] gives fit quantity for a single circuit, aggregated over outcomes.

    expected = (len(ds.outcome_labels) - 1)  # == "k"
    dof_per_box = expected; nboxes = len(circuit_list)
    pc = 0.05  # hardcoded (1 - confidence level) for now -- make into advanced option w/default

    circuit_weights = {}
    if scale_typ in ("robust", "Robust"):
        # Robust scaling V1: drastically scale down weights of especially bad sequences
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        for i, opstr in enumerate(circuit_list):
            if fitqty[i] > threshold:
                circuit_weights[opstr] = expected / fitqty[i]  # scaling factor

    elif scale_typ in ("robust+", "Robust+"):
        # Robust scaling V2: V1 + rescale to desired chi2 distribution without reordering
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        scaled_fitqty = fitqty.copy()
        for i, opstr in enumerate(circuit_list):
            if fitqty[i] > threshold:
                circuit_weights[opstr] = expected / fitqty[i]  # scaling factor
                scaled_fitqty[i] = expected  # (fitqty[i]*circuitWeights[opstr])

        nelements = len(fitqty)
        percentiles = [_chi2.ppf((i + 1) / (nelements + 1), dof_per_box) for i in range(nelements)]
        for ibin, i in enumerate(_np.argsort(scaled_fitqty)):
            opstr = circuit_list[i]
            fit, expected = scaled_fitqty[i], percentiles[ibin]
            if fit > expected:
                if opstr in circuit_weights: circuit_weights[opstr] *= expected / fit
                else: circuit_weights[opstr] = expected / fit

    return circuit_weights


def _compute_wildcard_budget(mdc_store, parameters, badfit_options, verbosity):
    """
    Create a wildcard budget for a model estimate.

    Parameters
    ----------
    model : Model
        The model to add a wildcard budget to.

    ds : DataSet
        The data the model predictions are being compared with.

    circuits_to_use : list
        The circuits whose data are compared.

    parameters : dict
        Various parameters of the estimate at hand.

    badfit_options : GSTBadFitOptions, optional
        Options specifying what post-processing actions should be performed when
        a fit is unsatisfactory.  Contains detailed parameters for wildcard budget
        creation.

    comm : mpi4py.MPI.Comm, optional
        An MPI communicator used to run this computation in parallel.

    mem_limit : int, optional
        A rough per-processor memory limit in bytes.

    verbosity : int, optional
        Level of detail printed to stdout.

    Returns
    -------
    PrimitiveOpsWildcardBudget
    """
    comm = mdc_store.resource_alloc.comm
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)
    fitqty = _get_fit_qty(mdc_store, parameters)
    badfit_options = GSTBadFitOptions.cast(badfit_options)
    model = mdc_store.model
    ds = mdc_store.dataset
    layout = mdc_store.layout
    circuits_to_use = layout.circuits  # (also mdc_store.circuits)

    printer.log("******************* Adding Wildcard Budget **************************")

    # Approach: we create an objective function that, for a given Wvec, computes:
    # (amt_of_2DLogL over threshold) + (amt of "red-box": per-outcome 2DlogL over threshold) + eta*|Wvec|_1                                     # noqa
    # and minimize this for different eta (binary search) to find that largest eta for which the
    # first two terms is are zero.  This Wvec is our keeper.

    ds_dof = ds.degrees_of_freedom(circuits_to_use)  # number of independent parameters
    # in dataset (max. model # of params)
    nparams = model.num_modeltest_params  # just use total number of params
    percentile = 0.025; nboxes = len(circuits_to_use)
    two_dlogl_threshold = _chi2.ppf(1 - percentile, ds_dof - nparams)
    redbox_threshold = _chi2.ppf(1 - percentile / nboxes, 1)
    # if p is prob that box is red and there are N boxes, then prob of no red boxes is q = (1-p)^N ~= 1-p*N
    # and so probability of any red boxes is ~p*N.  Here `percentile` is the probability of seeing *any* red
    # boxes, i.e. ~p*N, so to compute the prob of a single box being red we compute `p = percentile/N`.

    #print("DB2: ",twoDeltaLogL_threshold,redbox_threshold)

    objective = parameters.get('objective', 'logl')
    assert(objective == "logl"), "Can only use wildcard scaling with 'logl' objective!"
    two_dlogl_terms = fitqty
    two_dlogl = sum(two_dlogl_terms)

    if badfit_options.wildcard_initial_budget is None:
        primitive_op_labels = badfit_options.wildcard_primitive_op_labels
        if primitive_op_labels is None:
            primitive_op_labels = model.primitive_op_labels + model.primitive_instrument_labels
            if badfit_options.wildcard_budget_includes_spam:
                primitive_op_labels += ('SPAM',)  # special op name
        budget = _wild.PrimitiveOpsWildcardBudget(primitive_op_labels, start_budget=0.0)
    else:
        budget = badfit_options.wildcard_initial_budget

    ret = _collections.OrderedDict()
    zero_budget_is_ok = bool(two_dlogl <= two_dlogl_threshold
                             and sum(_np.clip(two_dlogl_terms - redbox_threshold, 0, None)) < 1e-6)
    if zero_budget_is_ok:
        printer.log("No need to add budget!")

    L1weights = _np.ones(budget.num_params)
    if badfit_options.wildcard_L1_weights:
        for op_label, weight in badfit_options.wildcard_L1_weights.items():
            L1weights[budget.primOpLookup[op_label]] = weight
        printer.log("Using non-uniform L1 weights: " + str(list(L1weights)))

    objfn_builder = parameters.get('final_objfn_builder', _objfns.PoissonPicDeltaLogLFunction.builder())
    objfn = objfn_builder.build_from_store(mdc_store)
    # assert this is a logl function?

    # Note: evaluate objfn before passing to wildcard fn init so internal probs are init
    dlogl_percircuit = objfn.percircuit(model.to_vector())
    assert(_np.allclose(two_dlogl_terms, 2 * dlogl_percircuit))  # should be equal: FUTURE: remove above call?

    logl_wildcard_fn = _objfns.LogLWildcardFunction(objfn, model.to_vector(), budget)
    wv_orig = budget.to_vector()

    for method_or_methodchain in badfit_options.wildcard_methods:
        methodchain = method_or_methodchain if isinstance(method_or_methodchain, (list, tuple)) \
            else (method_or_methodchain,)
        budget.from_vector(wv_orig)  # restore original budget to begin each chain
        chain_name = None
        budget_was_optimized = False
        tStart = _time.time()

        for method_name_or_dict in methodchain:
            if isinstance(method_name_or_dict, dict):
                method_name = method_name_or_dict.pop('name')
                method_options = method_name_or_dict
            else:
                method_name = method_name_or_dict
                method_options = {}

            #Update the name for the entire chain
            if 'chain_name' in method_options: chain_name = method_options.pop('chain_name')
            if chain_name is None: chain_name = method_name
            if method_name != "none": budget_was_optimized = True

            #Run the method
            if zero_budget_is_ok:  # no method needed - the zero budget meets constraints
                budget.from_vector(_np.zeros(len(budget.to_vector()), 'd'))

            elif method_name == "neldermead":
                _opt.optimize_wildcard_budget_neldermead(budget, L1weights, logl_wildcard_fn, layout,
                                                         two_dlogl_threshold, redbox_threshold, printer,
                                                         **method_options)
            elif method_name == "barrier":
                _opt.optimize_wildcard_budget_barrier(budget, L1weights, objfn, layout, two_dlogl_threshold,
                                                      redbox_threshold, printer, **method_options)
            elif method_name == "cvxopt":
                _opt.optimize_wildcard_budget_cvxopt(budget, L1weights, objfn, layout, two_dlogl_threshold,
                                                     redbox_threshold, printer, **method_options)
            elif method_name == "cvxopt_smoothed":
                _opt.optimize_wildcard_budget_cvxopt_smoothed(budget, L1weights, objfn, layout,
                                                              two_dlogl_threshold, redbox_threshold,
                                                              printer, **method_options)
            elif method_name == "cvxopt_small":
                _opt.optimize_wildcard_budget_cvxopt_zeroreg(budget, L1weights, objfn, layout,
                                                             two_dlogl_threshold, redbox_threshold, printer,
                                                             **method_options)
            elif method_name == "cvxpy_noagg":
                _opt.optimize_wildcard_budget_percircuit_only_cvxpy(budget, L1weights, objfn, layout,
                                                                    redbox_threshold, printer,
                                                                    **method_options)
            elif method_name == "none":
                pass
            else:
                raise ValueError("Invalid wildcard method name: %s" % method_name)

        #Done with chain: print result and check constraints
        def _evaluate_constraints(wv):
            dlogl_elements = logl_wildcard_fn.lsvec(wv)**2  # b/c WC fn only has sqrt of terms implemented now
            for i in range(len(layout.circuits)):
                dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)

            two_dlogl_percircuit = 2 * dlogl_percircuit
            two_dlogl = sum(two_dlogl_percircuit)
            return (max(0, two_dlogl - two_dlogl_threshold),
                    _np.clip(two_dlogl_percircuit - redbox_threshold, 0, None))

        wvec = budget.to_vector()
        wvec = _np.abs(wvec)
        budget.from_vector(wvec)  # ensure all budget elements are positive
        agg_constraint_violation, percircuit_constraint_violation = _evaluate_constraints(wvec)
        L1term = float(_np.sum(_np.abs(wvec) * L1weights))
        if chain_name is None: chain_name = "none"
        printer.log("Final wildcard budget for '%s' chain gives: (elapsed time %.1fs)" %
                    (chain_name, _time.time() - tStart))
        printer.log("   aggregate logl constraint violation = %g" % agg_constraint_violation)
        printer.log("   per-circuit logl constraint violation (totaled)= %g" % sum(percircuit_constraint_violation))
        printer.log("   L1-like term = %g" % L1term)
        printer.log("   " + str(budget))
        printer.log("")

        # Test that the found wildcard budget is admissable (there is not a strictly smaller wildcard budget
        # that also satisfies the constraints), and while doing this find the active constraints.
        printer.log("VERIFYING that the final wildcard budget vector is admissable")

        # Used for deciding what counts as a negligable per-gate wildcard.
        max_depth = 0
        for circ in ds.keys():
            if circ.depth > max_depth:
                max_depth = circ.depth

        active_constraints_list = []
        for w_ind, w_ele in enumerate(wvec):
            active_constraints = {}
            strictly_smaller_wvec = wvec.copy()
            negligable_budget = 1 / (100 * max_depth)
            if abs(w_ele) > negligable_budget:  # Use absolute values everywhere (wildcard vector can be negative).
                strictly_smaller_wvec[w_ind] = 0.99 * abs(w_ele)  # Decrease the vector element by 1%.
                printer.log(" - Trialing strictly smaller vector, with element %.3g reduced from %.3g to %.3g" %
                            (w_ind, w_ele, strictly_smaller_wvec[w_ind]))
                glob_constraint, percircuit_constraint = _evaluate_constraints(strictly_smaller_wvec)
                if glob_constraint + _np.sum(percircuit_constraint) < 1e-4:

                    toprint = ("   - Constraints still satisfied, budget NOT ADMISSABLE! Global = %.3g,"
                               " max per-circuit = %.3g ") % (glob_constraint, _np.max(percircuit_constraint))
                    # Throw an error if we are optimizing since this shouldn't happen then, otherwise just notify
                    if budget_was_optimized and badfit_options.wildcard_inadmissable_action == 'raise':
                        raise ValueError(toprint)
                    else:
                        printer.log(toprint)
                else:
                    printer.log(("   - Constraints (correctly) no longer satisfied! Global = %.3g, "
                                 "max per-circuit = %.3g ") % (glob_constraint, _np.max(percircuit_constraint)))

                circ_ind_max = _np.argmax(percircuit_constraint)
                if glob_constraint > 0:
                    active_constraints['global'] = glob_constraint,
                if percircuit_constraint[circ_ind_max] > 0:
                    active_constraints['percircuit'] = (circ_ind_max, layout.circuits[circ_ind_max],
                                                        percircuit_constraint[circ_ind_max])
            else:
                if budget_was_optimized:
                    printer.log((" - Element %.3g is %.3g. This is below %.3g, so trialing snapping to zero"
                                 " and updating.") % (w_ind, w_ele, negligable_budget))
                    strictly_smaller_wvec[w_ind] = 0.
                    glob_constraint, percircuit_constraint = _evaluate_constraints(strictly_smaller_wvec)
                    if glob_constraint + _np.sum(percircuit_constraint) < 1e-4:
                        printer.log("   - Snapping to zero accepted!")
                        wvec = strictly_smaller_wvec.copy()
                    else:
                        printer.log("   - Snapping to zero NOT accepted! Global = %.3g, max per-circuit = %.3g " %
                                    (glob_constraint, _np.max(percircuit_constraint)))
                else:
                    # We do this instead when we're not optimizing the budget, as otherwise we'd change the budget.
                    printer.log(" - Skipping trialing reducing element %.3g below %.3g, as it is less than %.3g" %
                                (w_ind, w_ele, negligable_budget))
            active_constraints_list.append(active_constraints)
        budget.from_vector(wvec)

        # Note: active_constraints_list is typically stored in parameters['unmodeled_error active constraints']
        # of the relevant Estimate object.
        primOp_labels = _collections.defaultdict(list)
        for lbl, i in budget.primOpLookup.items(): primOp_labels[i].append(str(lbl))
        for i, active_constraints in enumerate(active_constraints_list):
            if active_constraints:
                printer.log("** ACTIVE constraints for " + "--".join(primOp_labels[i]) + " **")
                if 'global' in active_constraints:
                    printer.log("   global constraint:" + str(active_constraints['global']))
                if 'percircuit' in active_constraints:
                    _, circuit, constraint_amt = active_constraints['percircuit']
                    printer.log("   per-circuit constraint:" + circuit.str + " = " + str(constraint_amt))
            else:
                printer.log("(no active constraints for " + "--".join(primOp_labels[i]) + ")")
        printer.log("")

        ret[chain_name] = (budget, active_constraints_list)

    return ret


def _reoptimize_with_weights(mdc_store, circuit_weights_dict, objfn_builder, optimizer, verbosity):
    """
    Re-optimize a model after data counts have been scaled by circuit weights.
    TODO: docstring

    Parameters
    ----------
    model : Model
        The model to re-optimize.

    ds : DataSet
        The data set to compare againts.

    circuit_list : list
        The circuits for which data and predictions should be compared.

    circuit_weights_dict : dict
        A dictionary of circuit weights, such as that returned by
        :function:`_compute_robust_scaling`, giving the data-count scaling factors.

    objfn_builder : ObjectiveFunctionBuilder
        The objective function (builder) that represents the final stage of
        optimization.  This defines what objective function is minimized in
        this re-optimization.

    optimizer : Optimizer
        The optimizer to use.

    resource_alloc : ResourceAllocation, optional
        What resources are available and how they should be distributed.

    verbosity : int, optional
        Level of detail printed to stdout.

    Returns
    -------
    Model
        The re-optimized model, potentially the *same* object as `model`.
    """
    ds = mdc_store.dataset
    circuit_list = mdc_store.circuits
    resource_alloc = mdc_store.resource_alloc

    printer = _objs.VerbosityPrinter.create_printer(verbosity)
    printer.log("--- Re-optimizing after robust data scaling ---")
    circuit_weights = _np.array([circuit_weights_dict.get(c, 1.0) for c in circuit_list], 'd')
    bulk_circuit_list = _CircuitList(circuit_list, circuit_weights=circuit_weights)
    opt_result, mdl_reopt = _alg.run_gst_fit_simple(ds, mdc_store.model.copy(), bulk_circuit_list, optimizer,
                                                    objfn_builder, resource_alloc, printer - 1)
    return mdl_reopt


class ModelEstimateResults(_proto.ProtocolResults):
    """
    A results object that holds model estimates.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.

    init_circuits : bool, optional
        Whether `self.circuit_lists` should be initialized or not.
        (In special cases, this can be disabled for speed.)

    Attributes
    ----------
    dataset : DataSet
        The underlying data set.
    """
    #Note: adds functionality to bare ProtocolResults object but *doesn't*
    #add additional data storage - all is still within same members,
    #even if this is is exposed differently.

    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None, quick_load=False):
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

        quick_load : bool, optional
            Setting this to True skips the loading of data and experiment-design
            components that may take a long time to load. This can be useful
            all the information of interest lies only within the results object.

        Returns
        -------
        ModelEstimateResults
        """
        ret = super().from_dir(dirname, name, preloaded_data, quick_load)  # loads members; doesn't make parent "links"
        for est in ret.estimates.values():
            est.parent = ret  # link estimate to parent results object
        return ret

    def __init__(self, data, protocol_instance, init_circuits=True):
        """
        Initialize an empty Results object.
        """
        super().__init__(data, protocol_instance)

        #Initialize some basic "results" by just exposing the circuit lists more directly
        circuit_lists = _collections.OrderedDict()

        if init_circuits:
            edesign = self.data.edesign
            if isinstance(edesign, _proto.CircuitListsDesign):
                circuit_lists['iteration'] = [_CircuitList.cast(cl) for cl in edesign.circuit_lists]

                #Set "Ls and germs" info if available
                if isinstance(edesign, StandardGSTDesign):
                    circuit_lists['prep fiducials'] = edesign.prep_fiducials
                    circuit_lists['meas fiducials'] = edesign.meas_fiducials
                    circuit_lists['germs'] = edesign.germs

            else:
                #Single iteration
                circuit_lists['iteration'] = [_CircuitList.cast(edesign.all_circuits_needing_data)]

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
        """
        The underlying data set.
        """
        return self.data.dataset

    def to_nameddict(self):
        """
        Convert these results into nested :class:`NamedDict` objects.

        Returns
        -------
        NamedDict
        """
        ret = _tools.NamedDict('Estimate', 'category')
        for k, v in self.estimates.items():
            ret[k] = v
        return ret

    def add_estimates(self, results, estimates_to_add=None):
        """
        Add some or all of the estimates from `results` to this `Results` object.

        Parameters
        ----------
        results : Results
            The object to import estimates from.  Note that this object must contain
            the same data set and gate sequence information as the importing object
            or an error is raised.

        estimates_to_add : list, optional
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
            if estimates_to_add is None or estimate_key in estimates_to_add:
                if estimate_key in self.estimates:
                    _warnings.warn("Re-initializing the %s estimate" % estimate_key
                                   + " of this Results object!  Usually you don't"
                                   + " want to do this.")
                self.estimates[estimate_key] = results.estimates[estimate_key]

    def rename_estimate(self, old_name, new_name):
        """
        Rename an estimate in this Results object.

        Ordering of estimates is not changed.

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
            models_by_iter = estimate.models.get('iteration estimates', [])
            la, lb = len(self.circuit_lists['iteration']), len(models_by_iter)
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

    def add_model_test(self, target_model, themodel,
                       estimate_key='test', gaugeopt_keys="auto"):
        """
        Add a new model-test (i.e. non-optimized) estimate to this `Results` object.

        Parameters
        ----------
        target_model : Model
            The target model used for comparison to the model.

        themodel : Model
            The "model" model whose fit to the data and distance from
            `target_model` are assessed.

        estimate_key : str, optional
            The key or label used to identify this estimate.

        gaugeopt_keys : list, optional
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
        # fill in what we can with info from existing estimates
        gaugeopt_suite = gaugeopt_target = None
        objfn_builder = None
        badfit_options = None
        for est in self.estimates.values():
            proto = est.parameters.get('protocol', None)
            if proto:
                if hasattr(proto, 'gaugeopt_suite') and hasattr(proto, 'gaugeopt_target'):
                    gaugeopt_suite = proto.gaugeopt_suite
                    gaugeopt_target = proto.gaugeopt_target
                if hasattr(proto, 'badfit_options'):
                    badfit_options = proto.badfit_options
            objfn_builder = est.parameters.get('final_objfn_builder', objfn_builder)

        from .modeltest import ModelTest as _ModelTest
        mdltest = _ModelTest(themodel, target_model, gaugeopt_suite, gaugeopt_target,
                             objfn_builder, badfit_options, name=estimate_key)
        test_result = mdltest.run(self.data)
        self.add_estimates(test_result)

    def view(self, estimate_keys, gaugeopt_keys=None):
        """
        Creates a shallow copy of this Results object containing only the given estimate.

        This function an also filter based on gauge-optimization keys, only keeping
        a subset of those available.

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
        view.circuit_lists = self.circuit_lists

        if isinstance(estimate_keys, str):
            estimate_keys = [estimate_keys]
        for ky in estimate_keys:
            if ky in self.estimates:
                view.estimates[ky] = self.estimates[ky].view(gaugeopt_keys, view)

        return view

    def copy(self):
        """
        Creates a copy of this :class:`ModelEstimateResults` object.

        Returns
        -------
        ModelEstimateResults
        """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        data = _proto.ProtocolData(self.data.edesign, self.data.dataset)
        cpy = ModelEstimateResults(data, self.protocol, init_circuits=False)
        cpy.circuit_lists = _copy.deepcopy(self.circuit_lists)
        for est_key, est in self.estimates.items():
            cpy.estimates[est_key] = est.copy()
        return cpy

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
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
