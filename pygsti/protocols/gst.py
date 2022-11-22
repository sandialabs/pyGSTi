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

import collections as _collections
import copy as _copy
import os as _os
import pickle as _pickle
import time as _time
import warnings as _warnings

import numpy as _np
from scipy.stats import chi2 as _chi2

from pygsti.baseobjs.profiler import DummyProfiler as _DummyProfiler
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.protocols.estimate import Estimate as _Estimate
from pygsti.protocols import protocol as _proto
from pygsti.protocols.modeltest import ModelTest as _ModelTest
from pygsti import algorithms as _alg
from pygsti import circuits as _circuits
from pygsti import io as _io
from pygsti import models as _models
from pygsti import optimize as _opt
from pygsti import tools as _tools
from pygsti import baseobjs as _baseobjs
from pygsti.processors import QuditProcessorSpec as _QuditProcessorSpec
from pygsti.modelmembers import operations as _op
from pygsti.models import Model as _Model
from pygsti.models.gaugegroup import GaugeGroup as _GaugeGroup, GaugeGroupElement as _GaugeGroupElement
from pygsti.objectivefns import objectivefns as _objfns, wildcardbudget as _wild
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.modelmembers import states as _states, povms as _povms
from pygsti.tools.legacytools import deprecate as _deprecated_fn


#For results object:
ROBUST_SUFFIX_LIST = [".robust", ".Robust", ".robust+", ".Robust+"]
DEFAULT_BAD_FIT_THRESHOLD = 2.0


class HasProcessorSpec(object):
    """
    Adds to an experiment design a `processor_spec` attribute

    Parameters
    ----------
    processorspec_filename_or_obj : QuditProcessorSpec or str
        The processor API used by this experiment design.
    """

    def __init__(self, processorspec_filename_or_obj):
        self.processor_spec = _load_pspec(processorspec_filename_or_obj) \
            if (processorspec_filename_or_obj is not None) else None
        self.auxfile_types['processor_spec'] = 'serialized-object'

    @_deprecated_fn('This function stub will be removed soon.')
    def create_target_model(self, gate_type='auto', prep_type='auto', povm_type='auto'):
        """
        Deprecated function.
        """
        raise NotImplementedError(("This function has been removed because is was an API hack.  To properly create"
                                   " a model from a processor spec, you should use one of the model creation functions"
                                   " in pygsti.models.modelconstruction"))


class GateSetTomographyDesign(_proto.CircuitListsDesign, HasProcessorSpec):
    """
    Minimal experiment design needed for GST

    Parameters
    ----------
    processorspec_filename_or_obj : QuditProcessorSpec or str
        The processor API used by this experiment design.

    circuit_lists : list or PlaquetteGridCircuitStructure
        A list whose elements are themselves lists of :class:`Circuit`
        objects, specifying the data that needs to be taken.  Alternatively,
        a single :class:`PlaquetteGridCircuitStructure` object containing
        a sequence of circuits lists, each at a different "x" value (usually
        the maximum circuit depth).

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

    def __init__(self, processorspec_filename_or_obj, circuit_lists, all_circuits_needing_data=None,
                 qubit_labels=None, nested=False, remove_duplicates=True):
        super().__init__(circuit_lists, all_circuits_needing_data, qubit_labels, nested, remove_duplicates)
        HasProcessorSpec.__init__(self, processorspec_filename_or_obj)


class StandardGSTDesign(GateSetTomographyDesign):
    """
    Standard GST experiment design consisting of germ-powers sandwiched between fiducials.

    Parameters
    ----------
    processorspec_filename_or_obj : QuditProcessorSpec or str
        The processor API used by this experiment design.

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

    def __init__(self, processorspec_filename_or_obj, prep_fiducial_list_or_filename, meas_fiducial_list_or_filename,
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
        processor_spec_or_model = _load_pspec_or_model(processorspec_filename_or_obj)
        lists = _circuits.create_lsgst_circuit_lists(
            processor_spec_or_model, self.prep_fiducials, self.meas_fiducials, self.germs,
            self.maxlengths, self.fiducial_pairs, self.truncation_method, self.nested,
            self.fpr_keep_fraction, self.fpr_keep_seed, self.include_lgst,
            self.aliases, self.circuit_rules, dscheck, action_if_missing,
            self.germ_length_limits, verbosity)
        #FUTURE: add support for "advanced options" (probably not in __init__ though?):
        # trunc_scheme=advancedOptions.get('truncScheme', "whole germ powers")

        try:
            processor_spec = processor_spec_or_model.create_processor_spec() \
                if isinstance(processor_spec_or_model, _Model) else processor_spec_or_model
        except Exception:
            _warnings.warn("Given model failed to create a processor spec for StdGST experiment design!")
            processor_spec = None  # allow this to bail out

        super().__init__(processor_spec, lists, None, qubit_labels, self.nested)
        self.auxfile_types['prep_fiducials'] = 'text-circuit-list'
        self.auxfile_types['meas_fiducials'] = 'text-circuit-list'
        self.auxfile_types['germs'] = 'text-circuit-list'
        self.auxfile_types['germ_length_limits'] = 'circuit-str-json'
        self.auxfile_types['fiducial_pairs'] = 'circuit-str-json'
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

        ret = StandardGSTDesign(self.processor_spec, self.prep_fiducials, self.meas_fiducials,
                                self.germs, max_lengths, gll, self.fiducial_pairs,
                                self.fpr_keep_fraction, self.fpr_keep_seed, self.include_lgst, self.nested,
                                self.circuit_rules, self.aliases, dscheck, action_if_missing, self.qubit_labels,
                                verbosity, add_default_protocol=False)

        #filter the circuit lists in `ret` using those in `self` (in case self includes only a subset of
        # the circuits dictated by the germs, fiducials, and  fidpairs).
        return ret.truncate_to_design(self)


class GSTInitialModel(_NicelySerializable):
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

    def __init__(self, model=None, target_model=None, starting_point=None, depolarize_start=0, randomize_start=0,
                 lgst_gaugeopt_tol=1e-6, contract_start_to_cptp=False):
        # Note: starting_point can be an initial model or string
        self.model = model
        self.target_model = target_model
        if starting_point is None:
            self.starting_point = "target" if (model is None) else "User-supplied-Model"
        else:
            self.starting_point = starting_point

        self.lgst_gaugeopt_tol = lgst_gaugeopt_tol
        self.contract_start_to_cptp = contract_start_to_cptp
        self.depolarize_start = depolarize_start
        self.randomize_start = randomize_start

    def retrieve_model(self, edesign, gaugeopt_target, dataset, comm):
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
        #Get starting point (model), which is used to compute other quantities
        # Note: should compute on rank 0 and distribute?
        starting_pt = self.starting_point
        if starting_pt == "User-supplied-Model":
            mdl_start = self.model

        elif starting_pt in ("LGST", "LGST-if-possible"):
            #lgst_advanced = advancedOptions.copy(); lgst_advanced.update({'estimateLabel': "LGST", 'onBadFit': []})

            if self.model is not None:
                mdl_start = self.model
            elif self.target_model is not None:
                mdl_start = self.target_model.copy()
            else:
                mdl_start = None

            if mdl_start is None:
                raise ValueError(("LGST requires a model. Specify an initial model or use an experiment"
                                  " design with a processor specification"))

            lgst = LGST(mdl_start,
                        gaugeopt_suite=GSTGaugeOptSuite(
                            gaugeopt_argument_dicts={'lgst_gaugeopt': {'tol': self.lgst_gaugeopt_tol}},
                            gaugeopt_target=gaugeopt_target),
                        badfit_options=None, name="LGST")

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
                    # mdl_start = mdl_start (either the target model or constructed from edesign pspec)

            if starting_pt == "LGST":
                lgst_results = lgst.run(lgst_data)
                mdl_start = lgst_results.estimates['LGST'].models['lgst_gaugeopt']

        elif starting_pt == "target":
            if self.target_model is not None:
                mdl_start = self.target_model.copy()
            else:
                raise ValueError("Starting point == 'target' and target model not specified!")
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

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'starting_point': self.starting_point,  # can be initial model? if so need to memoize...
                      'depolarize_start': self.depolarize_start,
                      'randomize_start': self.randomize_start,
                      'contract_start_to_cptp': self.contract_start_to_cptp,
                      'lgst_gaugeopt_tol': self.lgst_gaugeopt_tol,
                      'model': self.model.to_nice_serialization() if (self.model is not None) else None,
                      'target_model': (self.target_model.to_nice_serialization()
                                       if (self.target_model is not None) else None),
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        model = _Model.from_nice_serialization(state['model']) \
            if (state['model'] is not None) else None
        target_model = _Model.from_nice_serialization(state['target_model']) \
            if (state['target_model'] is not None) else None
        return cls(model, target_model, state['starting_point'], state['depolarize_start'],
                   state['randomize_start'], state['lgst_gaugeopt_tol'], state['contract_start_to_cptp'])


class GSTBadFitOptions(_NicelySerializable):
    """
    Options for post-processing a GST fit that was unsatisfactory.

    Parameters
    ----------
    threshold : float, optional
        A threshold, given in number-of-standard-deviations, below which a
        GST fit is considered satisfactory (and no "bad-fit" processing is needed).

    actions : tuple, optional
        Actions to take when a GST fit is unsatisfactory. Allowed actions include:
        - 'wildcard': Find an admissable wildcard model...
        - 'ddist_wildcard': Fits a single parameter wildcard model in which
          the amount of wildcard error added to an operation is proportional
          to the diamond distance between that operation and the target.
        - 'robust': scale data according out "robust statistics v1" algorithm,
           where we drastically scale down (reduce) the data due to especially
           poorly fitting circuits.  Namely, if a circuit's log-likelihood ratio
           exceeds the 95% confidence region about its expected value (the # of
           degrees of freedom in the circuits outcomes), then the data is scaled
           by the `expected_value / actual_value`, so that the new value exactly
           matches what would be expected.  Ideally there are only a few of these
           "outlier" circuits, which correspond errors in the measurement apparatus.
        - 'Robust': same as 'robust', but re-optimize the final objective function
           (usually the log-likelihood) after performing the scaling to get the
           final estimate.
        - 'robust+': scale data according out "robust statistics v2" algorithm,
           which performs the v1 algorithm (see 'robust' above) and then further
           rescales all the circuit data to achieve the desired chi2 distribution
           of per-circuit goodness-of-fit values *without reordering* these values.
        - 'Robust+': same as 'robust+', but re-optimize the final objective function
           (usually the log-likelihood) after performing the scaling to get the
           final estimate.
        - 'do nothing': do not perform any additional actions.  Used to help avoid
           the need for special cases when working with multiple types of bad-fit actions.

    wildcard_budget_includes_spam : bool, optional
        Include a SPAM budget within the wildcard budget used to process
        the `"wildcard"` action.

    wildcard_L1_weights :  np.array, optional
        An array of weights affecting the L1 penalty term used to select a feasible
        wildcard error vector `w_i` that minimizes `sum_i weight_i* |w_i|` (a weighted
        L1 norm).  Elements of this array must correspond to those of the wildcard budget
        being optimized, typically the primitive operations of the estimated model - but
        to get the order right you should specify `wildcard_primitive_op_labels` to be sure.
        If `None`, then all weights are assumed to be 1.

    wildcard_primitive_op_labels: list, optional
        The primitive operation labels used to construct the :class:`PrimitiveOpsWildcardBudget`
        that is optimized.  If `None`, equal to `model.primitive_op_labels + model.primitive_instrument_labels`
        where `model` is the estimated model, with `'SPAM'` at the end if `wildcard_budget_includes_spam`
        is True.  When specified, should contain a subset of the default values.

    wildcard_methods: tuple, optional
        A list of the methods to use to optimize the wildcard error vector.  Default is `("neldermead",)`.
        Options include `"neldermead"`, `"barrier"`, `"cvxopt"`, `"cvxopt_smoothed"`, `"cvxopt_small"`,
        and `"cvxpy_noagg"`.  So many methods exist because different convex solvers behave differently
        (unfortunately).  Leave as the default as a safe option, but `"barrier"` is pretty reliable and much
        faster than `"neldermead"`, and is a good option so long as it runs.

    wildcard_inadmissable_action: {"print", "raise"}, optional
        What to do when an inadmissable wildcard error vector is found.  The default just prints this
        information and continues, while `"raise"` raises a `ValueError`.  Often you just want this information
        printed so that when the wildcard analysis fails in this way it doesn't cause the rest of an analysis
        to abort.
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
                 wildcard_inadmissable_action='print', wildcard1d_reference='diamond distance'):
        valid_actions = ('wildcard', 'wildcard1d', 'Robust+', 'Robust', 'robust+', 'robust', 'do nothing')
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
        self.wildcard1d_reference = wildcard1d_reference

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'threshold': self.threshold,
                      'actions': self.actions,
                      'wildcard': {'budget_includes_spam': self.wildcard_budget_includes_spam,
                                   'L1_weights': self.wildcard_L1_weights,  # an array?
                                   'primitive_op_labels': self.wildcard_primitive_op_labels,
                                   'initial_budget': self.wildcard_initial_budget,  # serializable?
                                   'methods': self.wildcard_methods,
                                   'indadmissable_action': self.wildcard_inadmissable_action,
                                   '1d_reference': self.wildcard1d_reference},
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        wildcard = state.get('wildcard', {})
        cls(state['threshold'], tuple(state['actions']),
            wildcard.get('budget_includes_spam', True),
            wildcard.get('L1_weights', None),
            wildcard.get('primitive_op_labels', None),
            wildcard.get('initial_budget', None),
            tuple(wildcard.get('methods', ['neldermead'])),
            wildcard.get('inadmissable_action', 'print'),
            wildcard.get('1d_reference', 'diamond distance'))


class GSTObjFnBuilders(_NicelySerializable):
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

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({
            'iteration_builders': [b.to_nice_serialization() for b in self.iteration_builders],
            'final_builders': [b.to_nice_serialization() for b in self.final_builders]
        })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        iteration_builders = [_objfns.ObjectiveFunctionBuilder.from_nice_serialization(b)
                              for b in state['iteration_builders']]
        final_builders = [_objfns.ObjectiveFunctionBuilder.from_nice_serialization(b)
                          for b in state['final_builders']]
        return cls(iteration_builders, final_builders)


class GSTGaugeOptSuite(_NicelySerializable):
    """
    Holds directives to perform one or more gauge optimizations on a model.

    Usually this gauge optimization is done after fitting a parameterized
    model to data (e.g. after GST), as the data cannot (by definition)
    prefer any particular gauge choice.

    Parameters
    ----------
    gaugeopt_suite_names : str or list of strs, optional
        Names one or more gauge optimization suites to perform.  A string or
        list of strings (see below) specifies built-in sets of gauge optimizations.
        The built-in suites are:

          - "single" : performs only a single "best guess" gauge optimization.
          - "varySpam" : varies spam weight and toggles SPAM penalty (0 or 1).
          - "varySpamWt" : varies spam weight but no SPAM penalty.
          - "varyValidSpamWt" : varies spam weight with SPAM penalty == 1.
          - "toggleValidSpam" : toggles spame penalty (0 or 1); fixed SPAM wt.
          - "unreliable2Q" : adds branch to a spam suite that weights 2Q gates less
          - "none" : no gauge optimizations are performed.

    gaugeopt_argument_dicts : dict, optional
        A dictionary whose string-valued keys label different gauge optimizations (e.g. within a
        resulting `Estimate` object).  Each corresponding value can be either a dictionary
        of arguments to :func:`gaugeopt_to_target` or a list of such dictionaries which then
        describe the different stages of a multi-stage gauge optimization.

    gaugeopt_target : Model, optional
        If not None, a model to be used as the "target" for gauge-
        optimization (only).  This argument is useful when you want to
        gauge optimize toward something other than the *ideal* target gates
        given by the target model, which are used as the default when
        `gaugeopt_target` is None.
    """
    @classmethod
    def cast(cls, obj):
        if obj is None:
            return cls()  # None -> gaugeopt suite with default args (empty suite)
        elif isinstance(obj, GSTGaugeOptSuite):
            return obj
        elif isinstance(obj, (str, tuple, list)):
            return cls(gaugeopt_suite_names=obj)
        elif isinstance(obj, dict):
            return cls(gaugeopt_argument_dicts=obj)
        else:
            raise ValueError("Could not convert %s object to a gauge optimization suite!" % str(type(obj)))

    def __init__(self, gaugeopt_suite_names=None, gaugeopt_argument_dicts=None, gaugeopt_target=None):
        if gaugeopt_suite_names is not None:
            self.gaugeopt_suite_names = (gaugeopt_suite_names,) \
                if isinstance(gaugeopt_suite_names, str) else tuple(gaugeopt_suite_names)
        else:
            self.gaugeopt_suite_names = None

        if gaugeopt_argument_dicts is not None:
            self.gaugeopt_argument_dicts = gaugeopt_argument_dicts.copy()
        else:
            self.gaugeopt_argument_dicts = None

        self.gaugeopt_target = gaugeopt_target

    def is_empty(self):
        """
        Whether this suite is completely empty, i.e., contains NO gauge optimization instructions.

        This is a useful check before constructing quantities needed by gauge optimization,
        e.g. a target model, which can just be skipped when no gauge optimization will be performed.

        Returns
        -------
        bool
        """
        return (self.gaugeopt_suite_names is None) and (self.gaugeopt_argument_dicts is None)

    def to_dictionary(self, model, unreliable_ops=(), verbosity=0):
        """
        Converts this gauge optimization suite into a raw dictionary of dictionaries.

        Constructs a dictionary of gauge-optimization parameter dictionaries based
        on "gauge optimization suite" name(s).

        This essentially renders the gauge-optimization directives within this object
        in an "expanded" form for either running gauge optimization (e.g. within
        a :method:`GateSetTomography.run` call) or for constructing the would-be gauge
        optimization call arguments so they can be slightly modeified before passing
        them in as the actual gauge-optimization suite used in an analysis (the
        resulting dictionary can be used to initialize a new `GSTGaugeOptSuite` object
        via the `gaugeopt_argument_dicts` argument.

        Parameters
        ----------
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
        printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

        #Build ordered dict of gauge optimization parameters
        gaugeopt_suite_dict = _collections.OrderedDict()
        if self.gaugeopt_suite_names is not None:
            for gaugeopt_suite_name in self.gaugeopt_suite_names:
                self._update_gaugeopt_dict_from_suitename(gaugeopt_suite_dict, gaugeopt_suite_name,
                                                          gaugeopt_suite_name, model, unreliable_ops, printer)

        if self.gaugeopt_argument_dicts is not None:
            for lbl, goparams in self.gaugeopt_argument_dicts.items():

                if hasattr(goparams, 'keys'):  # goparams is a simple dict
                    gaugeopt_suite_dict[lbl] = goparams.copy()
                    gaugeopt_suite_dict[lbl].update({'verbosity': printer})
                else:  # assume goparams is an iterable
                    assert(isinstance(goparams, (list, tuple))), \
                        "If not a dictionary, gauge opt params should be a list or tuple of dicts!"
                    gaugeopt_suite_dict[lbl] = []
                    for goparams_stage in goparams:
                        dct = goparams_stage.copy()
                        dct.update({'verbosity': printer})
                        gaugeopt_suite_dict[lbl].append(dct)

        if self.gaugeopt_target is not None:
            assert(isinstance(self.gaugeopt_target, _Model)), "`gaugeopt_target` must be None or a Model"
            for goparams in gaugeopt_suite_dict.values():
                goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
                for goparams_dict in goparams_list:
                    if 'target_model' in goparams_dict:
                        _warnings.warn(("`gaugeOptTarget` argument is overriding"
                                        " user-defined target_model in gauge opt"
                                        " param dict(s)"))
                    goparams_dict.update({'target_model': self.gaugeopt_target})

        return gaugeopt_suite_dict

    def _update_gaugeopt_dict_from_suitename(self, gaugeopt_suite_dict, root_lbl, suite_name, model,
                                             unreliable_ops, printer):
        if suite_name in ("stdgaugeopt", "stdgaugeopt-unreliable2Q", "stdgaugeopt-tt", "stdgaugeopt-safe",
                          "stdgaugeopt-noconversion", "stdgaugeopt-noconversion-safe"):

            stages = []  # multi-stage gauge opt
            gg = model.default_gauge_group
            convert_to = {'to_type': "full TP", 'flatten_structure': True, 'set_default_gauge_group': True} \
                if ('noconversion' not in suite_name and gg.name not in ("Full", "TP")) else None

            if isinstance(gg, _models.gaugegroup.TrivialGaugeGroup) and convert_to is None:
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
                        'convert_model_to': convert_to,
                        'gates_metric': metric, 'spam_metric': metric,
                        'item_weights': {'gates': 1.0, 'spam': 0.0},
                        'gauge_group': _models.gaugegroup.UnitaryGaugeGroup(model.state_space,
                                                                            model.basis, model.evotype),
                        'oob_check_interval': 1 if ('-safe' in suite_name) else 0,
                        'verbosity': printer
                    })

                #Stage 3: spam gauge opt that fixes spam scaling at expense of
                #         non-unital parts of gates (but shouldn't affect these
                #         elements much since they should be small from Stage 2).
                s3gg = _models.gaugegroup.SpamGaugeGroup if (gg.name == "Full") else \
                    _models.gaugegroup.TPSpamGaugeGroup
                stages.append(
                    {
                        'convert_model_to': convert_to,
                        'gates_metric': metric, 'spam_metric': metric,
                        'item_weights': {'gates': 0.0, 'spam': 1.0},
                        'spam_penalty_factor': 1.0,
                        'gauge_group': s3gg(model.state_space, model.evotype),
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
            raise ValueError(("unreliable2Q is no longer a separate 'suite'.  You should precede it with the suite"
                              " name, e.g. 'stdgaugeopt-unreliable2Q' or 'varySpam-unreliable2Q'"))
        elif suite_name == "none":
            pass  # add nothing
        else:
            raise ValueError("Unknown gauge-optimization suite '%s'" % suite_name)

    def __getstate__(self):
        #Don't pickle comms in gaugeopt argument dicts
        to_pickle = self.__dict__.copy()
        if self.gaugeopt_argument_dicts is not None:
            to_pickle['gaugeopt_argument_dicts'] = _collections.OrderedDict()
            for lbl, goparams in self.gaugeopt_argument_dicts.items():
                if hasattr(goparams, "keys"):
                    if 'comm' in goparams:
                        goparams = goparams.copy()
                        goparams['comm'] = None
                    to_pickle['gaugeopt_argument_dicts'][lbl] = goparams
                else:  # goparams is a list
                    new_goparams = []  # new list
                    for goparams_dict in goparams:
                        if 'comm' in goparams_dict:
                            goparams_dict = goparams_dict.copy()
                            goparams_dict['comm'] = None
                        new_goparams.append(goparams_dict)
                    to_pickle['gaugeopt_argument_dicts'][lbl] = new_goparams
        return to_pickle

    def _to_nice_serialization(self):
        dicts_to_serialize = {}
        if self.gaugeopt_argument_dicts is not None:
            for lbl, goparams in self.gaugeopt_argument_dicts.items():
                goparams_list = [goparams] if hasattr(goparams, 'keys') else goparams
                serialize_list = []
                for goparams_dict in goparams_list:
                    to_add = goparams_dict.copy()
                    if 'target_model' in to_add:
                        to_add['target_model'] = goparams_dict['target_model'].to_nice_serialization()
                    if 'model' in to_add:
                        del to_add['model']  # don't serialize model argument
                    if 'comm' in to_add:
                        del to_add['comm']  # don't serialize comm argument
                    if '_gaugeGroupEl' in to_add:
                        to_add['_gaugeGroupEl'] = goparams_dict['_gaugeGroupEl'].to_nice_serialization()
                    if 'gauge_group' in to_add:
                        to_add['gauge_group'] = goparams_dict['gauge_group'].to_nice_serialization()
                    if 'verbosity' in to_add and isinstance(to_add['verbosity'], _baseobjs.VerbosityPrinter):
                        to_add['verbosity'] = goparams_dict['verbosity'].verbosity  # just save as an integer
                    serialize_list.append(to_add)
                dicts_to_serialize[lbl] = serialize_list  # Note: always a list, even when 1 element (simpler)

        target_to_serialize = self.gaugeopt_target.to_nice_serialization() \
            if (self.gaugeopt_target is not None) else None

        state = super()._to_nice_serialization()
        state.update({'gaugeopt_suite_names': self.gaugeopt_suite_names,
                      'gaugeopt_argument_dicts': dicts_to_serialize,
                      'gaugeopt_target': target_to_serialize
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        gaugeopt_argument_dicts = {}
        for lbl, serialized_goparams_list in state['gaugeopt_argument_dicts'].items():
            goparams_list = []
            for serialized_goparams in serialized_goparams_list:
                to_add = serialized_goparams.copy()
                if 'target_model' in to_add:
                    to_add['target_model'] = _Model.from_nice_serialization(serialized_goparams['target_model'])
                if '_gaugeGroupEl' in to_add:
                    to_add['_gaugeGroupEl'] = _GaugeGroupElement.from_nice_serialization(
                        serialized_goparams['_gaugeGroupEl'])
                if 'gauge_group' in to_add:
                    to_add['gauge_group'] = _GaugeGroup.from_nice_serialization(
                        serialized_goparams['gauge_group'])

                goparams_list.append(to_add)
            gaugeopt_argument_dicts[lbl] = goparams_list[0] if (len(goparams_list) == 1) else goparams_list

        if state['gaugeopt_target'] is not None:
            gaugeopt_target = _Model.from_nice_serialization(state['gaugeopt_target'])
        else:
            gaugeopt_target = None

        return cls(state['gaugeopt_suite_names'], gaugeopt_argument_dicts, gaugeopt_target)


class GateSetTomography(_proto.Protocol):
    """
    The core gate set tomography protocol, which optimizes a parameterized model to (best) fit a data set.

    Parameters
    ----------
    initial_model : Model or GSTInitialModel, optional
        The starting-point Model.

    gaugeopt_suite : GSTGaugeOptSuite, optional
        Specifies which gauge optimizations to perform on each estimate.  Can also
        be any object that can be cast to a :class:`GSTGaugeOptSuite` object, such
        as a string or list of strings (see below) specifying built-in sets of gauge
        optimizations.  This object also optionally stores an alternate target model
        for gauge optimization.  This model is used as the "target" for gauge-
        optimization (only), and is useful when you want to gauge optimize toward
        something other than the *ideal* target gates.

    objfn_builders : GSTObjFnBuilders, optional
        The objective function(s) to optimize.  Can also be anything that can
        be cast to a :class:`GSTObjFnBuilders` object.

    optimizer : Optimizer, optional
        The optimizer to use.  Can also be anything that can be cast to a :class:`Optimizer`.

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
                 objfn_builders=None, optimizer=None,
                 badfit_options=None, verbosity=2, name=None):
        super().__init__(name)
        self.initial_model = GSTInitialModel.cast(initial_model)
        self.gaugeopt_suite = GSTGaugeOptSuite.cast(gaugeopt_suite)
        self.badfit_options = GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        # compute, and set if needed, a default "first iteration # of fintite-difference iters"
        from pygsti.forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
        mdl = self.initial_model.model
        default_first_fditer = 1 if mdl and isinstance(mdl.sim, _MatrixFSim) else 0

        if isinstance(optimizer, _opt.Optimizer):
            self.optimizer = optimizer
            if isinstance(optimizer, _opt.CustomLMOptimizer) and optimizer.first_fditer is None:
                #special behavior: can set optimizer's first_fditer to `None` to mean "fill with default"
                self.optimizer = _copy.deepcopy(optimizer)  # don't mess with caller's optimizer
                self.optimizer.first_fditer = default_first_fditer
        else:
            if optimizer is None: optimizer = {}
            if 'first_fditer' not in optimizer:  # then add default first_fditer value
                optimizer['first_fditer'] = default_first_fditer
            self.optimizer = _opt.CustomLMOptimizer.cast(optimizer)

        self.objfn_builders = GSTObjFnBuilders.cast(objfn_builders)

        self.auxfile_types['initial_model'] = 'serialized-object'
        self.auxfile_types['badfit_options'] = 'serialized-object'
        self.auxfile_types['optimizer'] = 'serialized-object'
        self.auxfile_types['objfn_builders'] = 'serialized-object'
        self.auxfile_types['gaugeopt_suite'] = 'serialized-object'

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
        elif profile == 1: profiler = _baseobjs.Profiler(comm, False)
        elif profile == 2: profiler = _baseobjs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _baseobjs.VerbosityPrinter.create_printer(self.verbosity, comm)
        if self.record_output and not printer.is_recording():
            printer.start_recording()

        resource_alloc = _ResourceAllocation(comm, memlimit, profiler,
                                             distribute_method=self.distribute_method)
        if _tools.sharedmemtools.shared_mem_is_enabled():  # enable use of shared memory
            resource_alloc.build_hostcomms()  # signals that we want to use shared intra-host memory

        circuit_lists = data.edesign.circuit_lists
        aliases = circuit_lists[-1].op_label_aliases if isinstance(circuit_lists[-1], _CircuitList) else None
        ds = data.dataset

        if self.oplabel_aliases:  # override any other aliases with ones specifically given
            aliases = self.oplabel_aliases

        bulk_circuit_lists = [_CircuitList(lst, aliases, self.circuit_weights)
                              for lst in circuit_lists]

        tnxt = _time.time(); profiler.add_time('GST: loading', tref); tref = tnxt
        mdl_start = self.initial_model.retrieve_model(data.edesign, self.gaugeopt_suite.gaugeopt_target,
                                                      data.dataset, comm)

        tnxt = _time.time(); profiler.add_time('GST: Prep Initial seed', tref); tref = tnxt

        #Run Long-sequence GST on data
        mdl_lsgst_list, optimums_list, final_objfn = _alg.run_iterative_gst(
            ds, mdl_start, bulk_circuit_lists, self.optimizer,
            self.objfn_builders.iteration_builders, self.objfn_builders.final_builders,
            resource_alloc, printer)

        tnxt = _time.time(); profiler.add_time('GST: total iterative optimization', tref); tref = tnxt

        #set parameters
        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        parameters['final_objfn_builder'] = self.objfn_builders.final_builders[-1] \
            if len(self.objfn_builders.final_builders) > 0 else self.objfn_builders.iteration_builders[-1]
        parameters['final_objfn'] = final_objfn  # Final obj. function evaluated at best-fit point (cache too)
        parameters['final_mdc_store'] = final_objfn  # Final obj. function is also a "MDC store"
        parameters['profiler'] = profiler
        # Note: we associate 'final_cache' with the Estimate, which means we assume that *all*
        # of the models in the estimate can use same evaltree, have the same default prep/POVMs, etc.

        #TODO: add qtys about fit from optimums_list

        ret = ModelEstimateResults(data, self)

        #Set target model for post-processing -- assume that this is the initial model unless overridden by the
        # gauge opt suite
        if self.gaugeopt_suite.gaugeopt_target is not None:
            target_model = self.gaugeopt_suite.gaugeopt_target
        elif self.initial_model.target_model is not None:
            target_model = self.initial_model.target_model.copy()
        elif self.initial_model.model is not None and self.gaugeopt_suite.is_empty() is False:
            # when we desparately need a target model but none have been specifically given: use initial model
            target_model = self.initial_model.model.copy()
        else:
            target_model = None

        estimate = _Estimate.create_gst_estimate(ret, target_model, mdl_start, mdl_lsgst_list, parameters)
        ret.add_estimate(estimate, estimate_key=self.name)

        return _add_gaugeopt_and_badfit(ret, self.name, target_model,
                                        self.gaugeopt_suite, self.unreliable_ops,
                                        self.badfit_options, self.optimizer, resource_alloc, printer)


class LinearGateSetTomography(_proto.Protocol):
    """
    The linear gate set tomography protocol.

    Parameters
    ----------
    target_model : Model, optional
        The target (ideal) model.

    gaugeopt_suite : GSTGaugeOptSuite, optional
        Specifies which gauge optimizations to perform on each estimate.  Can also
        be any object that can be cast to a :class:`GSTGaugeOptSuite` object, such
        as a string or list of strings (see below) specifying built-in sets of gauge
        optimizations.  This object also optionally stores an alternate target model
        for gauge optimization.  This model is used as the "target" for gauge-
        optimization (only), and is useful when you want to gauge optimize toward
        something other than the *ideal* target gates.

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

    def __init__(self, target_model=None, gaugeopt_suite='stdgaugeopt',
                 badfit_options=None, verbosity=2, name=None):
        super().__init__(name)
        self.target_model = target_model
        self.gaugeopt_suite = GSTGaugeOptSuite.cast(gaugeopt_suite)
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

        if self.target_model is not None:
            target_model = self.target_model
        else:
            raise ValueError("LGST requires a target model and none was given!")

        if isinstance(target_model, _models.ExplicitOpModel):
            if not all([(isinstance(g, _op.FullArbitraryOp)
                         or isinstance(g, _op.FullTPOp))
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

        if self.target_model is not None:
            target_model = self.target_model
        else:
            raise ValueError("No target model specified.  Cannot run LGST.")

        if isinstance(edesign, _proto.CircuitListsDesign):
            circuit_list = edesign.circuit_lists[0]
        else:
            circuit_list = edesign.all_circuits_needing_data  # Never reached, since design must be a StandardGSTDesign!
        circuit_list = _CircuitList.cast(circuit_list)

        profile = self.profile
        if profile == 0: profiler = _DummyProfiler()
        elif profile == 1: profiler = _baseobjs.Profiler(comm, False)
        elif profile == 2: profiler = _baseobjs.Profiler(comm, True)
        else: raise ValueError("Invalid value for 'profile' argument (%s)" % profile)

        printer = _baseobjs.VerbosityPrinter.create_printer(self.verbosity, comm)
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
                                 op_labels, svd_truncate_to=target_model.state_space.dim,
                                 op_label_aliases=aliases, verbosity=printer)
        final_store = _objfns.ModelDatasetCircuitsStore(mdl_lgst, ds, circuit_list, resource_alloc,
                                                        array_types=('E',), verbosity=printer)

        parameters = _collections.OrderedDict()
        parameters['protocol'] = self  # Estimates can hold sub-Protocols <=> sub-results
        parameters['profiler'] = profiler
        parameters['final_mdc_store'] = final_store
        parameters['final_objfn_builder'] = _objfns.PoissonPicDeltaLogLFunction.builder()
        # just set final objective function as default logl objective (for ease of later comparison)

        ret = ModelEstimateResults(data, self)
        estimate = _Estimate(ret, {'target': target_model, 'seed': target_model, 'lgst': mdl_lgst,
                                   'iteration 0 estimate': mdl_lgst,
                                   'final iteration estimate': mdl_lgst},
                             parameters)
        ret.add_estimate(estimate, estimate_key=self.name)
        return _add_gaugeopt_and_badfit(ret, self.name, target_model, self.gaugeopt_suite,
                                        self.unreliable_ops, self.badfit_options,
                                        None, resource_alloc, printer)


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

    gaugeopt_suite : GSTGaugeOptSuite, optional
        Specifies which gauge optimizations to perform on each estimate.  Can also
        be any object that can be cast to a :class:`GSTGaugeOptSuite` object, such
        as a string or list of strings (see below) specifying built-in sets of gauge
        optimizations.  This object also optionally stores an alternate target model
        for gauge optimization.  This model is used as the "target" for gauge-
        optimization (only), and is useful when you want to gauge optimize toward
        something other than the *ideal* target gates.

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

    def __init__(self, modes="full TP,CPTP,Target", gaugeopt_suite='stdgaugeopt', target_model=None,
                 models_to_test=None, objfn_builders=None, optimizer=None, badfit_options=None, verbosity=2, name=None):

        super().__init__(name)
        self.modes = modes.split(',')
        self.models_to_test = models_to_test
        self.target_model = target_model
        self.gaugeopt_suite = GSTGaugeOptSuite.cast(gaugeopt_suite)
        self.objfn_builders = objfn_builders
        self.optimizer = _opt.CustomLMOptimizer.cast(optimizer)
        self.badfit_options = GSTBadFitOptions.cast(badfit_options)
        self.verbosity = verbosity

        if not isinstance(optimizer, _opt.Optimizer) and isinstance(optimizer, dict) \
           and 'first_fditer' not in optimizer:  # then a dict was cast into a CustomLMOptimizer above.
            # by default, set special "first_fditer=auto" behavior (see logic in GateSetTomography.__init__)
            self.optimizer.first_fditer = None

        self.auxfile_types['target_model'] = 'serialized-object'
        self.auxfile_types['models_to_test'] = 'dict:serialized-object'
        self.auxfile_types['gaugeopt_suite'] = 'serialized-object'
        self.auxfile_types['objfn_builders'] = 'serialized-object'
        self.auxfile_types['optimizer'] = 'serialized-object'
        self.auxfile_types['badfit_options'] = 'serialized-object'

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
        printer = _baseobjs.VerbosityPrinter.create_printer(self.verbosity, comm)

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

        if self.target_model is not None:
            target_model = self.target_model
        elif isinstance(data.edesign, HasProcessorSpec):
            # warnings.warn(...) -- or try/except and warn if fails?
            target_model = _models.modelconstruction._create_explicit_model(
                data.edesign.processor_spec, None, evotype='default', simulator='auto',
                ideal_gate_type='static', ideal_prep_type='auto', ideal_povm_type='auto',
                embed_gates=False, basis='pp')  # HARDCODED basis!

        ret = ModelEstimateResults(data, self)
        with printer.progress_logging(1):
            for i, mode in enumerate(modes):
                printer.show_progress(i, len(modes), prefix='-- Std Practice: ', suffix=' (%s) --' % mode)

                if mode == "Target":
                    model_to_test = target_model
                    mdltest = _ModelTest(model_to_test, target_model, self.gaugeopt_suite,
                                         mt_builder, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)

                elif mode in models_to_test:
                    mdltest = _ModelTest(models_to_test[mode], target_model, self.gaugeopt_suite,
                                         None, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = mdltest.run(data, memlimit, comm)
                    ret.add_estimates(result)

                else:
                    #Try to interpret `mode` as a parameterization
                    parameterization = mode  # for now, 1-1 correspondence
                    initial_model = target_model

                    try:
                        initial_model.set_all_parameterizations(parameterization)
                    except ValueError as e:
                        raise ValueError("Could not interpret '%s' mode as a parameterization! Details:\n%s"
                                         % (mode, str(e)))

                    initial_model = GSTInitialModel(initial_model, self.starting_point.get(mode, None))
                    gst = GST(initial_model, self.gaugeopt_suite, self.objfn_builders,
                              self.optimizer, self.badfit_options, verbosity=printer - 1, name=mode)
                    result = gst.run(data, memlimit, comm)
                    ret.add_estimates(result)

        return ret


# ------------------ HELPER FUNCTIONS -----------------------------------

def _load_pspec(processorspec_filename_or_obj):
    if not isinstance(processorspec_filename_or_obj, _QuditProcessorSpec):
        with open(processorspec_filename_or_obj, 'rb') as f:
            return _pickle.load(f)
    else:
        return processorspec_filename_or_obj


def _load_model(model_filename_or_obj):
    if isinstance(model_filename_or_obj, str):
        return _io.load_model(model_filename_or_obj)
    else:
        return model_filename_or_obj  # assume a Model object


def _load_pspec_or_model(processorspec_or_model_filename_or_obj):
    if isinstance(processorspec_or_model_filename_or_obj, str):
        # if a filename is given, just try to load a processor spec (can't load a model file yet)
        with open(processorspec_or_model_filename_or_obj, 'rb') as f:
            return _pickle.load(f)
    else:
        return processorspec_or_model_filename_or_obj


def _load_fiducials_and_germs(prep_fiducial_list_or_filename,
                              meas_fiducial_list_or_filename,
                              germ_list_or_filename):

    if isinstance(prep_fiducial_list_or_filename, str):
        prep_fiducials = _io.read_circuit_list(prep_fiducial_list_or_filename)
    else: prep_fiducials = prep_fiducial_list_or_filename

    if meas_fiducial_list_or_filename is None:
        meas_fiducials = prep_fiducials  # use same strings for meas_fiducials if meas_fiducial_list_or_filename is None
    else:
        if isinstance(meas_fiducial_list_or_filename, str):
            meas_fiducials = _io.read_circuit_list(meas_fiducial_list_or_filename)
        else: meas_fiducials = meas_fiducial_list_or_filename

    #Get/load germs
    if isinstance(germ_list_or_filename, str):
        germs = _io.read_circuit_list(germ_list_or_filename)
    else: germs = germ_list_or_filename

    return prep_fiducials, meas_fiducials, germs


def _load_dataset(data_filename_or_set, comm, verbosity):
    """Loads a DataSet from the data_filename_or_set argument of functions in this module."""
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    if isinstance(data_filename_or_set, str):
        if comm is None or comm.Get_rank() == 0:
            if _os.path.splitext(data_filename_or_set)[1] == ".pkl":
                with open(data_filename_or_set, 'rb') as pklfile:
                    ds = _pickle.load(pklfile)
            else:
                ds = _io.read_dataset(data_filename_or_set, True, "aggregate", printer)
            if comm is not None: comm.bcast(ds, root=0)
        else:
            ds = comm.bcast(None, root=0)
    else:
        ds = data_filename_or_set  # assume a Dataset object

    return ds


def _add_gaugeopt_and_badfit(results, estlbl, target_model, gaugeopt_suite,
                             unreliable_ops, badfit_options, optimizer, resource_alloc, printer):
    tref = _time.time()
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler

    #Do final gauge optimization to *final* iteration result only
    if gaugeopt_suite:
        model_to_gaugeopt = results.estimates[estlbl].models['final iteration estimate']
        if gaugeopt_suite.gaugeopt_target is None:  # add a default target model to gauge opt if needed
            #TODO: maybe make these two lines into a method of GSTGaugeOptSuite for adding a target model?
            gaugeopt_suite = _copy.deepcopy(gaugeopt_suite)
            gaugeopt_suite.gaugeopt_target = target_model
        _add_gauge_opt(results, estlbl, gaugeopt_suite,
                       model_to_gaugeopt, unreliable_ops, comm, printer - 1)
    profiler.add_time('%s: gauge optimization' % estlbl, tref); tref = _time.time()

    _add_badfit_estimates(results, estlbl, badfit_options, optimizer, resource_alloc, printer)
    profiler.add_time('%s: add badfit estimates' % estlbl, tref); tref = _time.time()

    #Add recorded info (even robust-related info) to the *base*
    #   estimate label's "stdout" meta information
    if printer.is_recording():
        results.estimates[estlbl].meta['stdout'] = printer.stop_recording()

    return results


def _add_gauge_opt(results, base_est_label, gaugeopt_suite, starting_model,
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

    gaugeopt_suite : GSTGaugeOptSuite, optional
        Specifies which gauge optimizations to perform on each estimate.  Can also
        be any object that can be cast to a :class:`GSTGaugeOptSuite` object, such
        as a string or list of strings (see below) specifying built-in sets of gauge
        optimizations.  This object also optionally stores an alternate target model
        for gauge optimization.  This model specifies the ideal gates and the default
        gauge group to optimize over (this should be set prior to calling this function).

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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)

    #Get gauge optimization dictionary
    gaugeopt_suite_dict = gaugeopt_suite.to_dictionary(starting_model,
                                                       unreliable_ops, printer - 1)

    #Gauge optimize to list of gauge optimization parameters
    for go_label, goparams in gaugeopt_suite_dict.items():

        printer.log("-- Performing '%s' gauge optimization on %s estimate --" % (go_label, base_est_label), 2)

        #Get starting model
        results.estimates[base_est_label].add_gaugeoptimized(goparams, None, go_label, comm, printer - 3)
        mdl_start = results.estimates[base_est_label].retrieve_start_model(goparams)

        #Gauge optimize data-scaled estimate also
        for suffix in ROBUST_SUFFIX_LIST:
            robust_est_label = base_est_label + suffix
            if robust_est_label in results.estimates:
                mdl_start_robust = results.estimates[robust_est_label].retrieve_start_model(goparams)

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
                          optimizer=None, resource_alloc=None, verbosity=0):
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

    if len(badfit_options.actions) == 0:
        return  # nothing to do - and exit before we try to evaluate objective fn.

    #Resource alloc gets sent to these estimate methods for building
    # a distributed-layout outjective fn / MDC store if one doesn't exist.
    mdc_objfn = base_estimate.final_objective_fn(resource_alloc)
    objfn_cache = base_estimate.final_objective_fn_cache(resource_alloc)  # ???

    ralloc = mdc_objfn.resource_alloc
    comm = ralloc.comm if ralloc else None
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, ralloc)

    if badfit_options.threshold is not None and \
       base_estimate.misfit_sigma(ralloc) <= badfit_options.threshold:
        return  # fit is good enough - no need to add any estimates

    assert(parameters.get('weights', None) is None), \
        "Cannot perform bad-fit scaling when weights are already given!"

    for badfit_typ in badfit_options.actions:
        new_params = parameters.copy()
        new_final_model = None

        if badfit_typ in ("robust", "Robust", "robust+", "Robust+"):
            global_weights = _compute_robust_scaling(badfit_typ, objfn_cache, mdc_objfn)
            new_params['weights'] = global_weights

            if badfit_typ in ("Robust", "Robust+") and (optimizer is not None) and (mdc_objfn is not None):
                mdc_objfn_reopt = _reoptimize_with_weights(mdc_objfn, global_weights, optimizer, printer - 1)
                new_final_model = mdc_objfn_reopt.model

        elif badfit_typ == "wildcard":
            try:
                budget_dict = _compute_wildcard_budget(objfn_cache, mdc_objfn, parameters, badfit_options, printer - 1)
                for chain_name, (unmodeled, active_constraint_list) in budget_dict.items():
                    base_estimate.extra_parameters[chain_name + "_unmodeled_error"] = unmodeled.to_nice_serialization()
                    base_estimate.extra_parameters[chain_name + "_unmodeled_active_constraints"] \
                        = active_constraint_list
                if len(budget_dict) > 0:  # also store first chain info w/empty chain name (convenience)
                    first_chain = next(iter(budget_dict))
                    unmodeled, active_constraint_list = budget_dict[first_chain]
                    base_estimate.extra_parameters["unmodeled_error"] = unmodeled.to_nice_serialization()
                    base_estimate.extra_parameters["unmodeled_active_constraints"] = active_constraint_list
            except NotImplementedError as e:
                printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                new_params['unmodeled_error'] = None
            #except AssertionError as e:
            #    printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
            #    new_params['unmodeled_error'] = None
            continue  # no need to add a new estimate - we just update the base estimate

        elif badfit_typ == 'wildcard1d':

            #If this estimate is the target model then skip adding the diamond distance wildcard.
            if base_estimate_label != 'Target':
                try:
                    budget = _compute_wildcard_budget_1d_model(base_estimate, objfn_cache, mdc_objfn, parameters,
                                                               badfit_options, printer - 1)

                    base_estimate.extra_parameters['wildcard1d' + "_unmodeled_error"] = budget.to_nice_serialization()
                    base_estimate.extra_parameters['wildcard1d' + "_unmodeled_active_constraints"] \
                        = None

                    base_estimate.extra_parameters["unmodeled_error"] = budget.to_nice_serialization()
                    base_estimate.extra_parameters["unmodeled_active_constraints"] = None
                except NotImplementedError as e:
                    printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                    new_params['unmodeled_error'] = None
                #except AssertionError as e:
                #    printer.warning("Failed to get wildcard budget - continuing anyway.  Error was:\n" + str(e))
                #    new_params['unmodeled_error'] = None
                continue  # no need to add a new estimate - we just update the base estimate

            else:
                printer.log('Diamond distance wildcard model is incompatible with the Target estimate, skipping.', 3)
                continue

        elif badfit_typ == "do nothing":
            continue  # go to next on-bad-fit directive

        else:
            raise ValueError("Invalid on-bad-fit directive: %s" % badfit_typ)

        # In case we've computed an updated final model, Just keep (?) old estimates of all
        # prior iterations (or use "blank" sentinel once this is supported).
        mdl_lsgst_list = [base_estimate.models['iteration %d estimate' % k]
                          for k in range(base_estimate.num_iterations)]
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
                _add_gauge_opt(results, base_estimate_label + '.' + badfit_typ,
                               GSTGaugeOptSuite(gaugeopt_argument_dicts={gokey: gauge_opt_params},
                                                gaugeopt_target=target_model),
                               new_final_model, unreliable_ops, comm, printer - 1)
            else:
                # add same gauge-optimized result as above
                go_gs_final = base_estimate.models[gokey]
                results.estimates[base_estimate_label + '.' + badfit_typ].add_gaugeoptimized(
                    gauge_opt_params.copy(), go_gs_final, gokey, comm, printer - 1)


def _compute_wildcard_budget_1d_model(estimate, objfn_cache, mdc_objfn, parameters, badfit_options, verbosity):
    """
    Create a wildcard budget for a model estimate. This version of the function produces a wildcard estimate
    using the model introduced by Tim and Stefan in the RCSGST paper.
    TODO: docstring (update)

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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, mdc_objfn.resource_alloc)
    badfit_options = GSTBadFitOptions.cast(badfit_options)
    model = mdc_objfn.model
    ds = mdc_objfn.dataset
    global_circuits_to_use = mdc_objfn.global_circuits

    printer.log("******************* Adding Wildcard Budget **************************")

    #cache construction code.
    # Extract model, dataset, objfn, etc.
    # Note: must evaluate mdc_objfn *before* passing to wildcard fn init so internal probs are init
    mdc_objfn.fn(model.to_vector())

    ## PYGSTI TRANSPLANT from pygsti.protocols.gst._compute_wildcard_budget
    # Compute the various thresholds
    ds_dof = ds.degrees_of_freedom(global_circuits_to_use)
    nparams = model.num_modeltest_params  # just use total number of params
    percentile = 0.025
    nboxes = len(global_circuits_to_use)

    two_dlogl_threshold = _chi2.ppf(1 - percentile, max(ds_dof - nparams, 1))
    redbox_threshold = _chi2.ppf(1 - percentile / nboxes, 1)

    ref, reference_name = _compute_1d_reference_values_and_name(estimate, badfit_options)
    primitive_ops = list(ref.keys())
    wcm = _wild.PrimitiveOpsSingleScaleWildcardBudget(primitive_ops, [ref[k] for k in primitive_ops],
                                                      reference_name=reference_name)
    _opt.wildcardopt.optimize_wildcard_bisect_alpha(wcm, mdc_objfn, two_dlogl_threshold, redbox_threshold, printer,
                                                    guess=0.1, tol=1e-3)  # results in optimized wcm
    return wcm


def _compute_1d_reference_values_and_name(estimate, badfit_options):
    final_model = estimate.models['final iteration estimate']
    target_model = estimate.models['target']
    gaugeopt_model = _alg.gaugeopt_to_target(final_model, target_model)

    if badfit_options.wildcard1d_reference == 'diamond distance':
        dd = {}
        for key, op in gaugeopt_model.operations.items():
            dd[key] = 0.5 * _tools.diamonddist(op.to_dense(), target_model.operations[key].to_dense())

        spamdd = {}
        for key, op in gaugeopt_model.preps.items():
            spamdd[key] = _tools.tracedist(_tools.vec_to_stdmx(op.to_dense(), 'pp'),
                                           _tools.vec_to_stdmx(target_model.preps[key].to_dense(), 'pp'))

        for key in gaugeopt_model.povms.keys():
            spamdd[key] = 0.5 * _tools.optools.povm_diamonddist(gaugeopt_model, target_model, key)

        dd['SPAM'] = sum(spamdd.values())
        return dd, 'diamond distance'
    else:
        raise ValueError("Invalid wildcard1d_reference value (%s) in bad-fit options!"
                         % str(badfit_options.wildcard1d_reference))


def _compute_robust_scaling(scale_typ, objfn_cache, mdc_objfn):
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
    #circuit_list = mdc_objfn.circuits  # *local* circuit list
    global_circuit_list = mdc_objfn.global_circuits  # *global* circuit list
    ds = mdc_objfn.dataset

    fitqty = objfn_cache.chi2k_distributed_percircuit  # *global* fit qty values (only for local circuits)
    #Note: fitqty[iCircuit] gives fit quantity for a single circuit, aggregated over outcomes.

    expected = (len(ds.outcome_labels) - 1)  # == "k"
    dof_per_box = expected; nboxes = len(global_circuit_list)
    pc = 0.05  # hardcoded (1 - confidence level) for now -- make into advanced option w/default

    circuit_weights = {}
    if scale_typ in ("robust", "Robust"):
        # Robust scaling V1: drastically scale down weights of especially bad sequences
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        for i, opstr in enumerate(global_circuit_list):
            if fitqty[i] > threshold:
                circuit_weights[opstr] = expected / fitqty[i]  # scaling factor

    elif scale_typ in ("robust+", "Robust+"):
        # Robust scaling V2: V1 + rescale to desired chi2 distribution without reordering
        threshold = _np.ceil(_chi2.ppf(1 - pc / nboxes, dof_per_box))
        scaled_fitqty = fitqty.copy()
        for i, opstr in enumerate(global_circuit_list):
            if fitqty[i] > threshold:
                circuit_weights[opstr] = expected / fitqty[i]  # scaling factor
                scaled_fitqty[i] = expected  # (fitqty[i]*circuitWeights[opstr])

        nelements = len(fitqty)
        percentiles = [_chi2.ppf((i + 1) / (nelements + 1), dof_per_box) for i in range(nelements)]
        for ibin, i in enumerate(_np.argsort(scaled_fitqty)):
            opstr = global_circuit_list[i]
            fit, expected = scaled_fitqty[i], percentiles[ibin]
            if fit > expected:
                if opstr in circuit_weights: circuit_weights[opstr] *= expected / fit
                else: circuit_weights[opstr] = expected / fit

    return circuit_weights  # contains *global* circuits as keys


def _compute_wildcard_budget(objfn_cache, mdc_objfn, parameters, badfit_options, verbosity):
    """
    Create a wildcard budget for a model estimate.
    TODO: update docstring

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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, mdc_objfn.resource_alloc)
    badfit_options = GSTBadFitOptions.cast(badfit_options)
    model = mdc_objfn.model
    ds = mdc_objfn.dataset
    global_circuits_to_use = mdc_objfn.global_circuits

    printer.log("******************* Adding Wildcard Budget **************************")

    # Approach: we create an objective function that, for a given Wvec, computes:
    # (amt_of_2DLogL over threshold) + (amt of "red-box": per-outcome 2DlogL over threshold) + eta*|Wvec|_1                                     # noqa
    # and minimize this for different eta (binary search) to find that largest eta for which the
    # first two terms is are zero.  This Wvec is our keeper.

    ds_dof = ds.degrees_of_freedom(global_circuits_to_use)  # number of independent parameters
    # in dataset (max. model # of params)
    nparams = model.num_modeltest_params  # just use total number of params
    percentile = 0.025; nboxes = len(global_circuits_to_use)
    if ds_dof < nparams:
        _warnings.warn(("Data has fewer degrees of freedom than model (%d < %d), and so to compute an "
                        "aggregate 2*DeltaLogL we'll use a k=1 chi2_k distribution. Please set "
                        "model.num_modeltest_params to an appropriate value!") % (ds_dof, nparams))

    two_dlogl_threshold = _chi2.ppf(1 - percentile, max(ds_dof - nparams, 1))
    redbox_threshold = _chi2.ppf(1 - percentile / nboxes, 1)
    # if p is prob that box is red and there are N boxes, then prob of no red boxes is q = (1-p)^N ~= 1-p*N
    # and so probability of any red boxes is ~p*N.  Here `percentile` is the probability of seeing *any* red
    # boxes, i.e. ~p*N, so to compute the prob of a single box being red we compute `p = percentile/N`.

    #print("DB2: ",twoDeltaLogL_threshold,redbox_threshold)

    assert(isinstance(mdc_objfn, _objfns.PoissonPicDeltaLogLFunction)), \
        "Can only use wildcard scaling with 'logl' objective!"

    two_dlogl_terms = objfn_cache.chi2k_distributed_percircuit  # *global* per-circuit 2*dlogl values
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
            L1weights[budget.primitive_op_param_index[op_label]] = weight
        printer.log("Using non-uniform L1 weights: " + str(list(L1weights)))

    # Note: must evaluate mdc_objfn *before* passing to wildcard fn init so internal probs are init
    mdc_objfn.fn(model.to_vector())
    logl_wildcard_fn = _objfns.LogLWildcardFunction(mdc_objfn, model.to_vector(), budget)

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
                _opt.optimize_wildcard_budget_neldermead(budget, L1weights, logl_wildcard_fn,
                                                         two_dlogl_threshold, redbox_threshold, printer,
                                                         **method_options)
            elif method_name == "barrier":
                _opt.optimize_wildcard_budget_barrier(budget, L1weights, mdc_objfn, two_dlogl_threshold,
                                                      redbox_threshold, printer, **method_options)
            elif method_name == "cvxopt":
                _opt.optimize_wildcard_budget_cvxopt(budget, L1weights, mdc_objfn, two_dlogl_threshold,
                                                     redbox_threshold, printer, **method_options)
            elif method_name == "cvxopt_smoothed":
                _opt.optimize_wildcard_budget_cvxopt_smoothed(budget, L1weights, mdc_objfn,
                                                              two_dlogl_threshold, redbox_threshold,
                                                              printer, **method_options)
            elif method_name == "cvxopt_small":
                _opt.optimize_wildcard_budget_cvxopt_zeroreg(budget, L1weights, mdc_objfn,
                                                             two_dlogl_threshold, redbox_threshold, printer,
                                                             **method_options)
            elif method_name == "cvxpy_noagg":
                _opt.optimize_wildcard_budget_percircuit_only_cvxpy(budget, L1weights, mdc_objfn,
                                                                    redbox_threshold, printer,
                                                                    **method_options)
            elif method_name == "none":
                pass
            else:
                raise ValueError("Invalid wildcard method name: %s" % method_name)

        #Done with chain: print result and check constraints
        def _evaluate_constraints(wv):
            layout = mdc_objfn.layout
            dlogl_elements = logl_wildcard_fn.lsvec(wv)**2  # b/c WC fn only has sqrt of terms implemented now
            dlogl_percircuit = _np.empty(len(layout.circuits), 'd')  # *local* circuits
            for i in range(len(layout.circuits)):
                dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)

            two_dlogl_percircuit = 2 * dlogl_percircuit
            two_dlogl = sum(two_dlogl_percircuit)
            global_two_dlogl_sum = layout.allsum_local_quantity('c', two_dlogl)
            global_two_dlogl_percircuit = layout.allgather_local_array('c', two_dlogl_percircuit)
            return (max(0, global_two_dlogl_sum - two_dlogl_threshold),
                    _np.clip(global_two_dlogl_percircuit - redbox_threshold, 0, None))

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
                    active_constraints['global'] = float(glob_constraint),
                if percircuit_constraint[circ_ind_max] > 0:
                    active_constraints['percircuit'] = (int(circ_ind_max), global_circuits_to_use[circ_ind_max].str,
                                                        float(percircuit_constraint[circ_ind_max]))
                #Note: make sure active_constraints is JSON serializable (this is why we put the circuit *str* in)
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
        for lbl, i in budget.primitive_op_param_index.items(): primOp_labels[i].append(str(lbl))
        for i, active_constraints in enumerate(active_constraints_list):
            if active_constraints:
                printer.log("** ACTIVE constraints for " + "--".join(primOp_labels[i]) + " **")
                if 'global' in active_constraints:
                    printer.log("   global constraint:" + str(active_constraints['global']))
                if 'percircuit' in active_constraints:
                    _, circuit_str, constraint_amt = active_constraints['percircuit']
                    printer.log("   per-circuit constraint:" + circuit_str + " = " + str(constraint_amt))
            else:
                printer.log("(no active constraints for " + "--".join(primOp_labels[i]) + ")")
        printer.log("")

        ret[chain_name] = (budget, active_constraints_list)

    return ret


def _reoptimize_with_weights(mdc_objfn, circuit_weights_dict, optimizer, verbosity):
    """
    Re-optimize a model after data counts have been scaled by circuit weights.
    TODO: update docstring

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
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    printer.log("--- Re-optimizing after robust data scaling ---")
    circuit_list = mdc_objfn.circuits
    circuit_weights = _np.array([circuit_weights_dict.get(c, 1.0) for c in circuit_list], 'd')

    # We update the circuit weights in mdc_objfn and reoptimize:
    orig_weights = mdc_objfn.circuits.circuit_weights
    mdc_objfn.circuits.circuit_weights = circuit_weights

    opt_result, mdc_objfn_reopt = _alg.run_gst_fit(mdc_objfn, optimizer, None, printer - 1)

    mdc_objfn.circuits.circuit_weights = orig_weights  # restore original circuit weights
    return mdc_objfn_reopt


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
        ret.circuit_lists = ret._create_circuit_lists(ret.data.edesign)  # because circuit_lists auxfile_type == 'none'
        for est in ret.estimates.values():
            est.parent = ret  # link estimate to parent results object
        return ret

    def __init__(self, data, protocol_instance, init_circuits=True):
        """
        Initialize an empty Results object.
        """
        super().__init__(data, protocol_instance)

        self.estimates = _collections.OrderedDict()
        self.circuit_lists = self._create_circuit_lists(self.data.edesign) \
            if init_circuits else _collections.OrderedDict()

        #Punt on serialization of these qtys for now...
        self.auxfile_types['circuit_lists'] = 'none'  # derived from edesign
        self.auxfile_types['estimates'] = 'dict:dir-serialized-object'

    def _create_circuit_lists(self, edesign):
        #Compute some basic "results" by just exposing edesign circuit lists more directly
        circuit_lists = _collections.OrderedDict()

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
        return circuit_lists

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
            la, lb = len(self.circuit_lists['iteration']), estimate.num_iterations
            assert(la == lb), "Number of iterations (%d) must equal %d!" % (lb, la)

        if estimate_key in self.estimates:
            _warnings.warn("Re-initializing the %s estimate" % estimate_key
                           + " of this Results object!  Usually you don't"
                           + " want to do this.")

        self.estimates[estimate_key] = estimate

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
        gaugeopt_suite = None
        objfn_builder = None
        badfit_options = None
        for est in self.estimates.values():
            proto = est.parameters.get('protocol', None)
            if proto:
                if hasattr(proto, 'gaugeopt_suite'):
                    gaugeopt_suite = proto.gaugeopt_suite
                if hasattr(proto, 'badfit_options'):
                    badfit_options = proto.badfit_options
            objfn_builder = est.parameters.get('final_objfn_builder', objfn_builder)

        from .modeltest import ModelTest as _ModelTest
        mdltest = _ModelTest(themodel, target_model, gaugeopt_suite,
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
            cpy.estimates[est_key].set_parent(cpy)
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
