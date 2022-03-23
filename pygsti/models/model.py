"""
Defines the Model class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import bisect as _bisect
import copy as _copy
import itertools as _itertools
import uuid as _uuid
import warnings as _warnings

import numpy as _np

from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.models.layerrules import LayerRules as _LayerRules
from pygsti.evotypes import Evotype as _Evotype
from pygsti.forwardsims import forwardsim as _fwdsim
from pygsti.forwardsims import mapforwardsim as _mapfwdsim
from pygsti.forwardsims import matrixforwardsim as _matrixfwdsim
from pygsti.modelmembers import modelmember as _gm
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.tools import slicetools as _slct
from pygsti.tools import matrixtools as _mt

MEMLIMIT_FOR_NONGAUGE_PARAMS = None


class Model(_NicelySerializable):
    """
    A predictive model for a Quantum Information Processor (QIP).

    The main function of a `Model` object is to compute the outcome
    probabilities of :class:`Circuit` objects based on the action of the
    model's ideal operations plus (potentially) noise which makes the
    outcome probabilities deviate from the perfect ones.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this model.
    """

    def __init__(self, state_space):
        self._state_space = _statespace.StateSpace.cast(state_space)
        self._num_modeltest_params = None
        self._hyperparams = {}
        self._paramvec = _np.zeros(0, 'd')
        self._paramlbls = _np.empty(0, dtype=object)
        self._param_bounds = None
        self.uuid = _uuid.uuid4()  # a Model's uuid is like a persistent id(), useful for hashing

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'state_space': self.state_space.to_nice_serialization()})
        return state

    @property
    def state_space(self):
        """
        State space labels

        Returns
        -------
        StateSpaceLabels
        """
        return self._state_space

    @property
    def hyperparams(self):
        """
        Dictionary of hyperparameters associated with this model

        Returns
        -------
        dict
        """
        return self._hyperparams  # Note: no need to set this param - just set/update values

    @property
    def num_params(self):
        """
        The number of free parameters when vectorizing this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        return len(self._paramvec)

    @property
    def num_modeltest_params(self):
        """
        The parameter count to use when testing this model against data.

        Often times, this is the same as :method:`num_params`, but there are times
        when it can convenient or necessary to use a parameter count different than
        the actual number of parameters in this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        if not hasattr(self, '_num_modeltest_params'):  # for backward compatibility
            self._num_modeltest_params = None

        if self._num_modeltest_params is not None:
            return self._num_modeltest_params
        elif 'num_nongauge_params' in dir(self):  # better than hasattr, which *runs* the @property method
            if MEMLIMIT_FOR_NONGAUGE_PARAMS is not None:
                if hasattr(self, 'num_elements'):
                    memForNumGaugeParams = self.num_elements * (self.num_params + self.state_space.dim**2) \
                        * _np.dtype('d').itemsize  # see Model._buildup_dpg (this is mem for dPG)
                else:
                    return self.num_params

                if memForNumGaugeParams > MEMLIMIT_FOR_NONGAUGE_PARAMS:
                    _warnings.warn(("Model.num_modeltest_params did not compute number of *non-gauge* parameters - "
                                    "using total (make MEMLIMIT_FOR_NONGAUGE_PARAMS larger if you really want "
                                    "the count of nongauge params"))
                    return self.num_params

            try:
                return self.num_nongauge_params  # len(x0)
            except:  # numpy can throw a LinAlgError or sparse cases can throw a NotImplementedError
                _warnings.warn(("Model.num_modeltest_params could not obtain number of *non-gauge* parameters"
                                " - using total instead"))
                return self.num_params
        else:
            return self.num_params

    @num_modeltest_params.setter
    def num_modeltest_params(self, count):
        self._num_modeltest_params = count

    @property
    def parameter_bounds(self):
        """ Upper and lower bounds on the values of each parameter, utilized by optimization routines """
        return self._param_bounds

    def set_parameter_bounds(self, index, lower_bound=-_np.inf, upper_bound=_np.inf):
        """
        Set the bounds for a single model parameter.

        These limit the values the parameter can have during an optimization of the model.

        Parameters
        ----------
        index : int
            The index of the paramter whose bounds should be set.

        lower_bound, upper_bound : float, optional
            The lower and upper bounds for the parameter.  Can be set to the special
            `numpy.inf` (or `-numpy.inf`) values to effectively have no bound.

        Returns
        -------
        None
        """
        if lower_bound == -_np.inf and upper_bound == _np.inf:
            return  # do nothing

        if self._param_bounds is None:
            self._param_bounds = _default_param_bounds(self.num_params)
        self._param_bounds[index, :] = (lower_bound, upper_bound)

    @property
    def parameter_labels(self):
        """
        A list of labels, usually of the form `(op_label, string_description)` describing this model's parameters.
        """
        return self._paramlbls

    @property
    def parameter_labels_pretty(self):
        """
        The list of parameter labels but formatted in a nice way.

        In particular, tuples where the first element is an op label are made into
        a single string beginning with the string representation of the operation.
        """
        ret = []
        for lbl in self.parameter_labels:
            if isinstance(lbl, (tuple, list)):
                ret.append(": ".join([str(x) for x in lbl]))
            else:
                ret.append(lbl)
        return ret

    def set_parameter_label(self, index, label):
        """
        Set the label of a single model parameter.

        Parameters
        ----------
        index : int
            The index of the paramter whose label should be set.

        label : object
            An object that serves to label this parameter.  Often a string.

        Returns
        -------
        None
        """
        self._paramlbls[index] = label

    def to_vector(self):
        """
        Returns the model vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        return self._paramvec

    def from_vector(self, v, close=False):
        """
        Sets this Model's operations based on parameter values `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters, with length equal to `self.num_params`.

        close : bool, optional
            Set to `True` if `v` is close to the current parameter vector.
            This can make some operations more efficient.

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params)
        self._paramvec = v.copy()

    def probabilities(self, circuit, clip_to=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        clip_to : 2-tuple, optional
            (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clip_to)
            for each spam label (string) SL.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def _init_copy(self, copy_into, memo):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """
        copy_into.uuid = _uuid.uuid4()  # new uuid for a copy (don't duplicate!)

    def _post_copy(self, copy_into, memo):
        """
        Called after all other copying is done, to perform "linking" between
        the new model (`copy_into`) and its members.
        """
        pass

    def copy(self):
        """
        Copy this model.

        Returns
        -------
        Model
            a (deep) copy of this model.
        """
        #Avoid having to reconstruct everything via __init__;
        # essentially deepcopy this object, but give the
        # class opportunity to initialize tricky members instead
        # of letting deepcopy do it.
        newModel = type(self).__new__(self.__class__)  # empty object

        memo = {}  # so that copying preserves linked object references

        #first call _init_copy to initialize any tricky members
        # (like those that contain references to self or other members)
        self._init_copy(newModel, memo)

        for attr, val in self.__dict__.items():
            if not hasattr(newModel, attr):
                assert(attr != "uuid"), "Should not be copying UUID!"
                setattr(newModel, attr, _copy.deepcopy(val, memo))

        self._post_copy(newModel, memo)
        return newModel

    def __str__(self):
        pass

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def circuit_outcomes(self, circuit):
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        return ()  # default = no outcomes

    def compute_num_outcomes(self, circuit):
        """
        The number of outcomes of `circuit`, given by it's existing or implied POVM label.

        Parameters
        ----------
        circuit : Circuit
            The circuit to simplify

        Returns
        -------
        int
        """
        return len(self.circuit_outcomes(circuit))

    def complete_circuit(self, circuit):
        """
        Adds any implied preparation or measurement layers to `circuit`

        Parameters
        ----------
        circuit : Circuit
            Circuit to act on.

        Returns
        -------
        Circuit
            Possibly the same object as `circuit`, if no additions are needed.
        """
        return circuit


class OpModel(Model):
    """
    A Model that contains operators (i.e. "members"), having a container structure.

    These operators are independently (sort of) parameterized and can be thought
    to have dense representations (even if they're not actually stored that way).
    This gives rise to the model having `basis` and `evotype` members.

    Secondly, attached to an `OpModel` is the idea of "circuit simplification"
    whereby the operators (preps, operations, povms, instruments) within
    a circuit get simplified to things corresponding to a single outcome
    probability, i.e. pseudo-circuits containing just preps, operations,
    and POMV effects.

    Thirdly, an `OpModel` is assumed to use a *layer-by-layer* evolution, and,
    because of circuit simplification process, the calculaton of circuit
    outcome probabilities has been pushed to a :class:`ForwardSimulator`
    object which just deals with the forward simulation of simplified circuits.
    Furthermore, instead of relying on a static set of operations a forward
    simulator queries a :class:`LayerLizard` for layer operations, making it
    possible to build up layer operations in an on-demand fashion from pieces
    within the model.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this model.

    basis : Basis
        The basis used for the state space by dense operator representations.

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    layer_rules : LayerRules
        The "layer rules" used for constructing operators for circuit
        layers.  This functionality is essential to using this model to
        simulate ciruits, and is typically supplied by derived classes.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The forward simulator (or typ) that this model should use.  `"auto"`
        tries to determine the best type automatically.
    """

    #Whether to perform extra parameter-vector integrity checks
    _pcheck = False

    #Experimental: whether to call .from_vector on operation *cache* elements as part of model.from_vector call
    _call_fromvector_on_cache = True

    def __init__(self, state_space, basis, evotype, layer_rules, simulator="auto"):
        """
        Creates a new OpModel.  Rarely used except from derived classes `__init__` functions.
        """
        self._evotype = _Evotype.cast(evotype)
        self._set_state_space(state_space, basis)
        #sets self._state_space, self._basis

        super(OpModel, self).__init__(self.state_space)  # do this as soon as possible

        self._layer_rules = layer_rules if (layer_rules is not None) else _LayerRules()
        self._opcaches = {}  # dicts of non-primitive operations (organized by derived class)
        self._need_to_rebuild = True  # whether we call _rebuild_paramvec() in to_vector() or num_params()
        self.dirty = False  # indicates when objects and _paramvec may be out of sync
        self.sim = simulator  # property setter does nontrivial initialization (do this *last*)
        self._param_interposer = None
        self._reinit_opcaches()

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self._sim.model = self  # ensure the simulator's `model` is set to self (usually == None in serialization)

    ##########################################
    ## Get/Set methods
    ##########################################

    @property
    def sim(self):
        """ Forward simulator for this model """
        self._clean_paramvec()  # clear opcache and rebuild paramvec when needed
        if hasattr(self._sim, 'model'):
            assert(self._sim.model is self), "Simulator out of sync with model!!"
        return self._sim

    @sim.setter
    def sim(self, simulator):
        try:  # don't fail if state space doesn't have an integral # of qubits
            nqubits = self.state_space.num_qubits
        except:
            nqubits = None
        # TODO: This should probably also take evotype (e.g. 'chp' should probably use a CHPForwardSim, etc)
        self._sim = simulator = _fwdsim.ForwardSimulator.cast(simulator, nqubits)
        self._sim.model = self  # ensure the simulator's `model` is set to this object

    @property
    def evotype(self):
        """
        Evolution type

        Returns
        -------
        str
        """
        return self._evotype

    @property
    def basis(self):
        """
        The basis used to represent dense (super)operators of this model

        Returns
        -------
        Basis
        """
        return self._basis

    @basis.setter
    def basis(self, basis):
        """
        The basis used to represent dense (super)operators of this model
        """
        if isinstance(basis, _Basis):
            assert(basis.is_compatible_with_state_space(self.state_space)), "Basis is incompabtible with state space!"
            self._basis = basis
        else:  # create a basis with the proper structure & dimension
            self._basis = _Basis.cast(basis, self.state_space)

    def _set_state_space(self, lbls, basis="pp"):
        """
        Sets labels for the components of the Hilbert space upon which the gates of this Model act.

        Parameters
        ----------
        lbls : list or tuple or StateSpaceLabels object
            A list of state-space labels (can be strings or integers), e.g.
            `['Q0','Q1']` or a :class:`StateSpaceLabels` object.

        basis : Basis or str
            A :class:`Basis` object or a basis name (like `"pp"`), specifying
            the basis used to interpret the operators in this Model.  If a
            `Basis` object, then its dimensions must match those of `lbls`.

        Returns
        -------
        None
        """
        if isinstance(lbls, _statespace.StateSpace):
            self._state_space = lbls
        else:
            #Maybe change to a different default?
            self._state_space = _statespace.ExplicitStateSpace(lbls)
        self.basis = basis  # invokes basis setter to set self._basis

        #Operator dimension of this Model
        #self._dim = self.state_space.dim
        #e.g. 4 for 1Q (densitymx) or 2 for 1Q (statevec)

    #TODO - deprecate this?
    @property
    def dim(self):
        """
        The dimension of the model.

        This equals d when the gate (or, more generally, circuit-layer) matrices
        would have shape d x d and spam vectors would have shape d x 1 (if they
        were computed).

        Returns
        -------
        int
            model dimension
        """
        return self._state_space.dim

    ####################################################
    ## Parameter vector maintenance
    ####################################################

    @property
    def num_params(self):
        """
        The number of free parameters when vectorizing this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        self._clean_paramvec()
        return len(self._paramvec)

    def _iter_parameterized_objs(self):
        raise NotImplementedError("Derived Model classes should implement _iter_parameterized_objs")
        #return # default is to have no parameterized objects

    def _check_paramvec(self, debug=False):
        if debug: print("---- Model._check_paramvec ----")

        TOL = 1e-8
        for lbl, obj in self._iter_parameterized_objs():
            if debug: print(lbl, ":", obj.num_params, obj.gpindices)
            w = obj.to_vector()
            msg = "None" if (obj.parent is None) else id(obj.parent)
            assert(obj.parent is self), "%s's parent is not set correctly (%s)!" % (lbl, msg)
            if obj.gpindices is not None and len(w) > 0:
                if _np.linalg.norm(self._paramvec[obj.gpindices] - w) > TOL:
                    if debug: print(lbl, ".to_vector() = ", w, " but Model's paramvec = ",
                                    self._paramvec[obj.gpindices])
                    raise ValueError("%s is out of sync with paramvec!!!" % lbl)
            if not self.dirty and obj.dirty:
                raise ValueError("%s is dirty but Model.dirty=False!!" % lbl)

    def _clean_paramvec(self):
        """ Updates _paramvec corresponding to any "dirty" elements, which may
            have been modified without out knowing, leaving _paramvec out of
            sync with the element's internal data.  It *may* be necessary
            to resolve conflicts where multiple dirty elements want different
            values for a single parameter.  This method is used as a safety net
            that tries to insure _paramvec & Model elements are consistent
            before their use."""

        #Note on dirty flag processing and the "dirty_value" flag of members:
        #    A model member's "dirty" flag is set to True when the member's
        #    value (local parameter vector) may be different from its parent
        #    model's parameter vector.  Usually, when `from_vector` is called on
        #    a member, this should set the dirty flag (since it sets the local
        #    parameter vector).  The exception is when this function is being
        #    called within the parent's `from_vector` method, in which case the
        #    flag should be reset to `False`, even if it was True before.
        #    Whether this operation should refrain from setting it's dirty
        #    flag as a result of this call.  `False` is the safe option, as
        #    this call potentially changes this operation's parameters.

        #print("Cleaning Paramvec (dirty=%s, rebuild=%s)" % (self.dirty, self._need_to_rebuild))
        #import inspect, pprint
        #pprint.pprint([(x.filename,x.lineno,x.function) for x in inspect.stack()[0:7]])

        if self._need_to_rebuild:
            self._rebuild_paramvec()
            self._need_to_rebuild = False
            self._reinit_opcaches()  # changes to parameter vector structure invalidate cached ops

        if self.dirty:  # if any member object is dirty (ModelMember.dirty setter should set this value)
            TOL = 1e-8
            ops_paramvec = self._model_paramvec_to_ops_paramvec(self._paramvec)

            #Note: lbl args used *just* for potential debugging - could strip out once
            # we're confident this code always works.
            def clean_single_obj(obj, lbl):  # sync an object's to_vector result w/_paramvec
                if obj.dirty:
                    w = obj.to_vector()
                    chk_norm = _np.linalg.norm(ops_paramvec[obj.gpindices] - w)
                    #print(lbl, " is dirty! vec = ", w, "  chk_norm = ",chk_norm)
                    if (not _np.isfinite(chk_norm)) or chk_norm > TOL:
                        ops_paramvec[obj.gpindices] = w
                    obj.dirty = False

            def clean_obj(obj, lbl):  # recursive so works with objects that have sub-members
                for i, subm in enumerate(obj.submembers()):
                    clean_obj(subm, _Label(lbl.name + ":%d" % i, lbl.sslbls))
                clean_single_obj(obj, lbl)

            for lbl, obj in self._iter_parameterized_objs():
                clean_obj(obj, lbl)

            #re-update everything to ensure consistency ~ self.from_vector(self._paramvec)
            #print("DEBUG: non-trivially CLEANED paramvec due to dirty elements")
            for _, obj in self._iter_parameterized_objs():
                obj.from_vector(ops_paramvec[obj.gpindices], dirty_value=False)
                #object is known to be consistent with _paramvec

            # Call from_vector on elements of the cache
            if self._call_fromvector_on_cache:
                for opcache in self._opcaches.values():
                    for obj in opcache.values():
                        obj.from_vector(ops_paramvec[obj.gpindices], dirty_value=False)

            self.dirty = False
            self._paramvec[:] = self._ops_paramvec_to_model_paramvec(ops_paramvec)
            #self._reinit_opcaches()  # this shouldn't be necessary

        if OpModel._pcheck: self._check_paramvec()

    def _mark_for_rebuild(self, modified_obj=None):
        #re-initialze any members that also depend on the updated parameters
        self._need_to_rebuild = True

        # Specifically, we need to re-allocate indices for every object that
        # contains a reference to the modified one.  Previously all modelmembers
        # of a model needed to have their self.parent *always* point to the
        # parent model, and so we would clear the members' .gpindices to indicate
        # allocation was needed.  Now, that constraint has been loosened, and
        # we instead set a member's .parent=None to indicate it needs reallocation
        # (this allows the .gpindices to be used, e.g., for parameter counting
        # within the object).  Because _rebuild_paramvec determines whether an
        # object needs allocation by calling its .gpindices_are_allocated method,
        # which checks submembers too, there's no longer any need to clear (set
        # to None) the .gpindinces any objects here.

        #OLD
        #for _, o in self._iter_parameterized_objs():
        #    if o._obj_refcount(modified_obj) > 0:
        #        o.clear_gpindices()  # ~ o.gpindices = None but works w/submembers
        #        # (so params for this obj will be rebuilt)

        self.dirty = True
        #since it's likely we'll set at least one of our object's .dirty flags
        # to True (and said object may have parent=None and so won't
        # auto-propagate up to set this model's dirty flag (self.dirty)

    def _print_gpindices(self, max_depth=100):
        print("PRINTING MODEL GPINDICES!!!")
        for lbl, obj in self._iter_parameterized_objs():
            obj._print_gpindices("", str(lbl), max_depth=max_depth)

    def print_parameters_by_op(self, max_depth=0):
        plbls = {i: lbl for i, lbl in enumerate(self.parameter_labels)}
        print("*** MODEL PARAMETERS (%d total) ***" % self.num_params)
        for lbl, obj in self._iter_parameterized_objs():
            obj._print_gpindices("", lbl, plbls, max_depth)

    def collect_parameters(self, params_to_collect, new_param_label=None):
        """
        Updates this model's parameters so that previously independent parameters are tied together.

        The model's parameterization is modified so that all of the parameters
        given by `params_to_collect` are replaced by a single parameter.  The label
        of this single parameter may be given if desired.

        Note that after this function is called the model's parameter vector (i.e. the
        result of `to_vector()`) should be assumed to have a new format unrelated to the
        parameter vector before their adjustment.  For example, you should not assume that
        un-modified parameters will retain their old indices.

        Parameters
        ----------
        params_to_collect : iterable
            A list or tuple of parameter labels describing the parameters to collect.
            These should be a subset of the elements of `self.parameter_labels` or
            of `self.parameter_labels_pretty`, or integer indices into the model's parameter
            vector.  If empty, no parameter adjustment is performed.

        new_param_label : object, optional
            The label for the new common parameter.  If `None`, then the parameter label
            of the first collected parameter is used.

        Returns
        -------
        None
        """
        if all([isinstance(p, int) for p in params_to_collect]):
            indices = list(params_to_collect)  # all parameters are given as indices
        else:
            plbl_dict = {plbl: i for i, plbl in enumerate(self.parameter_labels)}
            try:
                indices = [plbl_dict[plbl] for plbl in params_to_collect]
            except KeyError:
                plbl_dict_pretty = {plbl: i for i, plbl in enumerate(self.parameter_labels_pretty)}
                indices = [plbl_dict_pretty[plbl] for plbl in params_to_collect]
        indices.sort()

        if len(indices) == 0:
            return  # nothing to do

        # go through all gpindices and reset so that all occurrences of elements in
        # indices[1:] are updated to be indices[0]
        memo = set()  # keep track of which object's gpindices have been set
        for _, obj in self._iter_parameterized_objs():
            assert(obj.gpindices is not None and obj.parent is self), \
                "Model's parameter vector still needs to be built!"

            new_gpindices = None
            if isinstance(obj.gpindices, slice):
                if indices[0] >= obj.gpindices.stop or indices[-1] < obj.gpindices.start:
                    continue  # short circuit so we don't have to check condition in line below
                if any([obj.gpindices.start <= i < obj.gpindices.stop for i in indices[1:]]):
                    new_gpindices = obj.gpindices_as_array()
            else:
                if indices[0] > max(obj.gpindices) or indices[-1] < min(obj.gpindices):
                    continue  # short circuit
                new_gpindices = obj.gpindices.copy()

            if new_gpindices is not None:
                for k in indices[1:]:
                    new_gpindices[new_gpindices == k] = indices[0]
                obj.set_gpindices(new_gpindices, self, memo)

        #Rename the collected parameter if desired.
        if new_param_label is not None:
            self._paramlbls[indices[0]] = new_param_label

        # now all gpindices are updated, so just rebuild paramvec to remove unused indices.
        self._rebuild_paramvec()

    def uncollect_parameters(self, param_to_uncollect):
        """
        Updates this model's parameters so that a common paramter becomes independent parameters.

        The model's parameterization is modified so that each usage of the given parameter
        in the model's parameterized operations is promoted to being a new independent
        parameter. The labels of the new parameters are set by the operations.

        Note that after this function is called the model's parameter vector (i.e. the
        result of `to_vector()`) should be assumed to have a new format unrelated to the
        parameter vector before their adjustment.  For example, you should not assume that
        un-modified parameters will retain their old indices.

        Parameters
        ----------
        param_to_uncollect : int or object
            A parameter label specifying the parameter to "uncollect".  This should be an
            element of `self.parameter_labels` or `self.parameter_labels_pretty`, or it may be
            an integer index into the model's parameter vector.

        Returns
        -------
        None
        """
        if isinstance(param_to_uncollect, int):
            index = param_to_uncollect
        else:
            plbl_dict = {plbl: i for i, plbl in enumerate(self.parameter_labels)}
            try:
                index = plbl_dict[param_to_uncollect]
            except KeyError:
                plbl_dict_pretty = {plbl: i for i, plbl in enumerate(self.parameter_labels_pretty)}
                index = plbl_dict_pretty[param_to_uncollect]

        # go through all gpindices and reset so that each occurrence of `index` after the
        # first gets a new index => independent parameter.
        next_new_index = self.num_params; first_occurrence = True
        memo = set()  # keep track of which object's gpindices have been set
        for lbl, obj in self._iter_parameterized_objs():
            assert(obj.gpindices is not None and obj.parent is self), \
                "Model's parameter vector still needs to be built!"

            if id(obj) in memo:
                continue  # don't add any new indices when set_gpindices doesn't actually do anything

            new_gpindices = None
            if isinstance(obj.gpindices, slice):
                if obj.gpindices.start <= index < obj.gpindices.stop:
                    if first_occurrence:  # just update/reset parameter label
                        self._paramlbls[index] = (lbl, obj.parameter_labels[index - obj.gpindices.start])
                        first_occurrence = False
                    else:  # index as a new parameter
                        new_gpindices = obj.gpindices_as_array()
                        new_gpindices[index - obj.gpindices.start] = next_new_index
                        next_new_index += 1
            else:
                if min(obj.gpindices) <= index <= max(obj.gpindices):
                    new_gpindices = obj.gpindices.copy()
                    for i in range(len(new_gpindices)):
                        if new_gpindices[i] == index:
                            if first_occurrence:  # just update/reset parameter label
                                self._paramlbls[index] = (lbl, obj.parameter_labels[i])
                                first_occurrence = False
                            else:  # index as a new parameter
                                new_gpindices[i] = next_new_index
                                next_new_index += 1

            if new_gpindices is not None:
                obj.set_gpindices(new_gpindices, self, memo)

        # now all gpindices are updated, so just rebuild paramvec create new parameters.
        self._rebuild_paramvec()

    def _rebuild_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        w = self._model_paramvec_to_ops_paramvec(self._paramvec)
        Np = len(w)  # NOT self.num_params since the latter calls us!
        wl = self._paramlbls
        wb = self._param_bounds if (self._param_bounds is not None) else _default_param_bounds(Np)
        #NOTE: interposer doesn't quite work with parameter bounds yet, as we need to convert "model"
        # bounds to "ops" bounds like we do the parameter vector.  Need something like:
        #wb = self._model_parambouds_to_ops_parambounds(self._param_bounds) \
        #    if (self._param_bounds is not None) else _default_param_bounds(Np)
        debug = False
        if debug: print("DEBUG: rebuilding model %s..." % str(id(self)))

        # Step 1: add parameters that don't exist yet
        #  Note that iteration order (that of _iter_parameterized_objs) determines
        #  parameter index ordering, so "normally" an object that occurs before
        #  another in the iteration order will have gpindices which are lower - and
        #  when new indices are allocated we try to maintain this normal order by
        #  inserting them at an appropriate place in the parameter vector.

        #Get a record up-front, before any allocations are made, of which objects will need to be reallocated.
        is_allocated = {lbl: obj.gpindices_are_allocated(self) for lbl, obj in self._iter_parameterized_objs()}
        max_index_processed_so_far = -1

        for lbl, obj in self._iter_parameterized_objs():

            completely_allocated = is_allocated[lbl]
            if debug: print("Processing: ", lbl, " gpindices=", obj.gpindices, " allocated = ", completely_allocated)

            if not completely_allocated:  # obj.gpindices_are_allocated(self):
                # We need to [re-]allocate obj's indices to this model
                num_new_params, max_existing_index = obj.preallocate_gpindices(self)  # any new indices need allocation?
                max_index_processed_so_far = max(max_index_processed_so_far, max_existing_index)
                insertion_point = max_index_processed_so_far + 1
                if num_new_params > 0:
                    # If so, before allocating anything, make the necessary space in the parameter arrays:
                    for _, o in self._iter_parameterized_objs():
                        o.shift_gpindices(insertion_point, num_new_params, self)
                    w = _np.insert(w, insertion_point, _np.empty(num_new_params, 'd'))
                    wl = _np.insert(wl, insertion_point, _np.empty(num_new_params, dtype=object))
                    wb = _np.insert(wb, insertion_point, _default_param_bounds(num_new_params), axis=0)

                # Now allocate (actually updates obj's gpindices).  May be necessary even if
                # num_new_params == 0 (e.g. a composed op needs to have it's gpindices updated bc
                # a sub-member's # of params was updated, but this submember was already allocated
                # on an earlier iteration - this is why we compute is_allocated as outset).
                num_added_params = obj.allocate_gpindices(insertion_point, self)
                assert(num_added_params == num_new_params), \
                    "Inconsistency between preallocate_gpindices and allocate_gpindices!"
                if debug:
                    print("DEBUG: allocated %d new params starting at %d, resulting gpindices = %s"
                          % (num_new_params, insertion_point, str(obj.gpindices)))

                newly_added_indices = slice(insertion_point, insertion_point + num_added_params) \
                    if num_added_params > 0 else None  # for updating parameter labels below

            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds) > 0 else -1; L = len(w)
                if debug: print("DEBUG: %s: existing indices = " % (str(lbl)), obj.gpindices, " M=", M, " L=", L)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.

                    #set newly_added_indices to the indices for *obj* that have just been added
                    added_indices = slice(len(w), len(w) + (M + 1 - L))
                    if isinstance(obj.gpindices, slice):
                        newly_added_indices = _slct.intersect(added_indices, obj.gpindices)
                    else:
                        newly_added_indices = inds[_np.logical_and(inds >= added_indices.start,
                                                                   inds < added_indices.stop)]

                    w = _np.concatenate((w, _np.empty(M + 1 - L, 'd')), axis=0)  # [v.resize(M+1) doesn't work]
                    wl = _np.concatenate((wl, _np.empty(M + 1 - L, dtype=object)), axis=0)
                    wb = _np.concatenate((wb, _np.empty((M + 1 - L, 2), 'd')), axis=0)
                    if debug: print("DEBUG:    --> added %d new params" % (M + 1 - L))
                else:
                    newly_added_indices = None
                #if M >= 0:  # M == -1 signifies this object has no parameters, so we'll just leave `off` alone
                #    off = M + 1

            #Update max_index_processed_so_far
            max_gpindex = (obj.gpindices.stop - 1) if isinstance(obj.gpindices, slice) else max(obj.gpindices)
            max_index_processed_so_far = max(max_index_processed_so_far, max_gpindex)

            # Update parameter values / labels / bounds.
            # - updating the labels is not strictly necessary since we usually *clean* the param vector next
            # - we *always* sync object parameter bounds in case any have changed (when bounds on members are
            #   set/modified, model should be marked for rebuilding)
            # - we only update *new* parameter labels, since we want any modified labels in the model to "stick"
            w[obj.gpindices] = obj.to_vector()
            wb[obj.gpindices, :] = obj.parameter_bounds if (obj.parameter_bounds is not None) \
                else _default_param_bounds(obj.num_params)
            if newly_added_indices is not None:
                obj_paramlbls = _np.empty(obj.num_params, dtype=object)
                obj_paramlbls[:] = [(lbl, obj_plbl) for obj_plbl in obj.parameter_labels]
                wl[newly_added_indices] = obj_paramlbls[_gm._decompose_gpindices(obj.gpindices, newly_added_indices)]

        #Step 2: remove any unused indices from paramvec and shift accordingly
        used_gpindices = set()
        for lbl, obj in self._iter_parameterized_objs():
            #print("Removal: ",lbl,str(type(obj)),(id(obj.parent) if obj.parent is not None else None),obj.gpindices)
            assert(obj.parent is self and obj.gpindices is not None)
            used_gpindices.update(obj.gpindices_as_array())

            #OLD: from when this was step 1:
            #if obj.gpindices is not None:
            #    if obj.parent is self:  # then obj.gpindices lays claim to our parameters
            #        used_gpindices.update(obj.gpindices_as_array())
            #    else:
            #        # ok for objects to have parent=None (before their params are allocated), in which case non-None
            #        # gpindices can enable the object to function without a parent model, but not parent=other_model,
            #        # as this indicates the objects parameters are allocated to another model (and should have been
            #        # cleared in the OrderedMemberDict.__setitem__ method used to add the model member.
            #        assert(obj.parent is None), \
            #            "Member's parent (%s) is not set correctly! (must be this model or None)" % repr(obj.parent)
            #else:
            #    assert(obj.parent is self or obj.parent is None)
            #    #Note: ok for objects to have parent == None and gpindices == None

        Np = len(w)  # reset Np from possible new params (NOT self.num_params since the latter calls us!)
        indices_to_remove = sorted(set(range(Np)) - used_gpindices)
        if debug: print("Indices to remove = ", indices_to_remove, " of ", Np)

        if len(indices_to_remove) > 0:
            #if debug: print("DEBUG: Removing %d params:"  % len(indices_to_remove), indices_to_remove)
            w = _np.delete(w, indices_to_remove)
            wl = _np.delete(wl, indices_to_remove)
            wb = _np.delete(wb, indices_to_remove, axis=0)
            def _get_shift(j): return _bisect.bisect_left(indices_to_remove, j)
            memo = set()  # keep track of which object's gpindices have been set
            for _, obj in self._iter_parameterized_objs():
                # ensure object is allocated to this model and thus should be shifted:
                assert(obj.gpindices is not None and obj.parent is self)
                if id(obj) in memo: continue  # already processed
                if isinstance(obj.gpindices, slice):
                    new_inds = _slct.shift(obj.gpindices,
                                           -_get_shift(obj.gpindices.start))
                else:
                    new_inds = []
                    for i in obj.gpindices:
                        new_inds.append(i - _get_shift(i))
                    new_inds = _np.array(new_inds, _np.int64)
                obj.set_gpindices(new_inds, self, memo)

        self._paramvec = self._ops_paramvec_to_model_paramvec(w)
        self._paramlbls = self._ops_paramlbls_to_model_paramlbls(wl)
        self._param_bounds = wb if _param_bounds_are_nontrivial(wb) else None
        if debug: print("DEBUG: Done rebuild: %d op params" % len(w))

    def _init_virtual_obj(self, obj):
        """
        Initializes a "virtual object" - an object (e.g. LinearOperator) that *could* be a
        member of the Model but won't be, as it's just built for temporary
        use (e.g. the parallel action of several "base" gates).  As such
        we need to fully initialize its parent and gpindices members so it
        knows it belongs to this Model BUT it's not allowed to add any new
        parameters (they'd just be temporary).  It's also assumed that virtual
        objects don't need to be to/from-vectored as there are already enough
        real (non-virtual) gates/spamvecs/etc. to accomplish this.
        """
        if obj.gpindices is not None:
            assert(obj.parent is self or obj.parent is None), "Virtual obj has incorrect parent already set!"
            return  # if parent is already set we assume obj has already been init

        #Assume all parameters of obj are new independent parameters
        num_new_params = obj.allocate_gpindices(self.num_params, self)
        assert(num_new_params == 0), "Virtual object is requesting %d new params!" % num_new_params

    def _obj_refcount(self, obj):
        """ Number of references to `obj` contained within this Model """
        cnt = 0
        for _, o in self._iter_parameterized_objs():
            cnt += o._obj_refcount(obj)
        return cnt

    def to_vector(self):
        """
        Returns the model vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        self._clean_paramvec()  # will rebuild if needed
        return self._paramvec

    def from_vector(self, v, close=False):
        """
        Sets this Model's operations based on parameter values `v`.

        The inverse of to_vector.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters, with length equal to `self.num_params`.

        close : bool, optional
            Set to `True` if `v` is close to the current parameter vector.
            This can make some operations more efficient.

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params)

        self._paramvec = v.copy()
        w = self._model_paramvec_to_ops_paramvec(v)
        for _, obj in self._iter_parameterized_objs():
            obj.from_vector(w[obj.gpindices], close, dirty_value=False)
            # dirty_value=False => obj.dirty = False b/c object is known to be consistent with _paramvec

        # Call from_vector on elements of the cache
        if self._call_fromvector_on_cache:
            for opcache in self._opcaches.values():
                for obj in opcache.values():
                    obj.from_vector(w[obj.gpindices], close, dirty_value=False)

        if OpModel._pcheck: self._check_paramvec()

    @property
    def param_interposer(self):
        return self._param_interposer

    @param_interposer.setter
    def param_interposer(self, interposer):
        if self._param_interposer is not None:  # remove existing interposer
            self._paramvec = self._model_paramvec_to_ops_paramvec(self._paramvec)
        self._param_interposer = interposer
        if interposer is not None:  # add new interposer
            self._clean_paramvec()
            self._paramvec = self._ops_paramvec_to_model_paramvec(self._paramvec)

    def _model_paramvec_to_ops_paramvec(self, v):
        return self.param_interposer.model_paramvec_to_ops_paramvec(v) \
            if (self.param_interposer is not None) else v

    def _ops_paramvec_to_model_paramvec(self, w):
        return self.param_interposer.ops_paramvec_to_model_paramvec(w) \
            if (self.param_interposer is not None) else w

    def _ops_paramlbls_to_model_paramlbls(self, w):
        return self.param_interposer.ops_paramlbls_to_model_paramlbls(w) \
            if (self.param_interposer is not None) else w

    def circuit_outcomes(self, circuit):
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        outcomes = circuit.expand_instruments_and_separate_povm(self)  # dict w/keys=sep-povm-circuits, vals=outcomes
        return tuple(_itertools.chain(*outcomes.values()))  # concatenate outputs from all sep-povm-circuits

    def split_circuit(self, circuit, erroron=('prep', 'povm'), split_prep=True, split_povm=True):
        """
        Splits a circuit into prep_layer + op_layers + povm_layer components.

        If `circuit` does not contain a prep label or a
        povm label a default label is returned if one exists.

        Parameters
        ----------
        circuit : Circuit
            A circuit, possibly beginning with a state preparation
            label and ending with a povm label.

        erroron : tuple of {'prep','povm'}
            A ValueError is raised if a preparation or povm label cannot be
            resolved when 'prep' or 'povm' is included in 'erroron'.  Otherwise
            `None` is returned in place of unresolvable labels.  An exception
            is when this model has no preps or povms, in which case `None`
            is always returned and errors are never raised, since in this
            case one usually doesn't expect to use the Model to compute
            probabilities (e.g. in germ selection).

        split_prep : bool, optional
            Whether to split off the state prep and return it as `prep_label`.  If
            `False`, then the returned preparation label is always `None`, and is
            not removed from `ops_only_circuit`.

        split_povm : bool, optional
            Whether to split off the POVM and return it as `povm_label`.  If
            `False`, then the returned POVM label is always `None`, and is
            not removed from `ops_only_circuit`.

        Returns
        -------
        prep_label : str or None
        ops_only_circuit : Circuit
        povm_label : str or None
        """
        if split_prep:
            if len(circuit) > 0 and self._is_primitive_prep_layer_lbl(circuit[0]):
                prep_lbl = circuit[0]
                circuit = circuit[1:]
            elif self._default_primitive_prep_layer_lbl() is not None:
                prep_lbl = self._default_primitive_prep_layer_lbl()
            else:
                if 'prep' in erroron and self._has_primitive_preps():
                    raise ValueError("Cannot resolve state prep in %s" % circuit)
                else: prep_lbl = None
        else:
            prep_lbl = None

        if split_povm:
            if len(circuit) > 0 and self._is_primitive_povm_layer_lbl(circuit[-1]):
                povm_lbl = circuit[-1]
                circuit = circuit[:-1]
            elif self._default_primitive_povm_layer_lbl(circuit.line_labels) is not None:
                povm_lbl = self._default_primitive_povm_layer_lbl(circuit.line_labels)
            else:
                if 'povm' in erroron and self._has_primitive_povms():
                    raise ValueError("Cannot resolve POVM in %s" % str(circuit))
                else: povm_lbl = None
        else:
            povm_lbl = None

        return prep_lbl, circuit, povm_lbl

    def complete_circuit(self, circuit):
        """
        Adds any implied preparation or measurement layers to `circuit`

        Converts `circuit` into a "complete circuit", where the first (0-th)
        layer is a state preparation and the final layer is a measurement (POVM) layer.

        Parameters
        ----------
        circuit : Circuit
            Circuit to act on.

        Returns
        -------
        Circuit
            Possibly the same object as `circuit`, if no additions are needed.
        """
        prep_lbl_to_prepend = None
        povm_lbl_to_append = None

        if len(circuit) == 0 or not self._is_primitive_prep_layer_lbl(circuit[0]):
            prep_lbl_to_prepend = self._default_primitive_prep_layer_lbl()
            if prep_lbl_to_prepend is None:
                #raise ValueError(f"Missing state prep in {circuit.str} and there's no default!")
                raise ValueError("Missing state prep in %s and there's no default!" % circuit.str)

        if len(circuit) == 0 or not self._is_primitive_povm_layer_lbl(circuit[-1]):
            sslbls = circuit.line_labels if circuit.line_labels != ("*",) else None
            povm_lbl_to_append = self._default_primitive_povm_layer_lbl(sslbls)

            if povm_lbl_to_append is None:
                #raise ValueError(f"Missing POVM in {circuit.str} and there's no default!")
                raise ValueError("Missing POVM in %s and there's no default!" % circuit.str)

        if prep_lbl_to_prepend or povm_lbl_to_append:
            #SLOW way:
            #circuit = circuit.copy(editable=True)
            #if prep_lbl_to_prepend: circuit.insert_layer_inplace(prep_lbl_to_prepend, 0)
            #if povm_lbl_to_append: circuit.insert_layer_inplace(povm_lbl_to_append, len(circuit))
            #circuit.done_editing()
            if prep_lbl_to_prepend: circuit = (prep_lbl_to_prepend,) + circuit
            if povm_lbl_to_append: circuit = circuit + (povm_lbl_to_append,)

        return circuit

    # ---- Operation container interface ----
    # These functions allow oracle access to whether a label of a given type
    # "exists" (or can be created by) this model.

    # Support notion of "primitive" *layer* operations, which are
    # stored somewhere in this model (and so don't need to be cached)
    # and represent the fundamental building blocks of other layer operations.
    # "Primitive" layers are used to <TODO>

    # These properties should return an OrdereDict, whose keys can
    # be used as an ordered set (values can be anything - we don't care).
    @property
    def _primitive_prep_label_dict(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_povm_labels_dict(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_op_labels_dict(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_instrument_labels_dict(self):
        raise NotImplementedError("Derived classes must implement this!")

    # These are the public properties that return tuples
    @property
    def primitive_prep_labels(self):
        return tuple(self._primitive_prep_label_dict.keys())

    @property
    def primitive_povm_labels(self):
        return tuple(self._primitive_povm_label_dict.keys())

    @property
    def primitive_op_labels(self):
        return tuple(self._primitive_op_label_dict.keys())

    @property
    def primitive_instrument_labels(self):
        return tuple(self._primitive_instrument_label_dict.keys())

    def _is_primitive_prep_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid state prep label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_prep_label_dict

    def _is_primitive_povm_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid POVM label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_povm_label_dict or lbl.name in self._primitive_povm_label_dict

    def _is_primitive_op_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid operation label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_op_label_dict

    def _is_primitive_instrument_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid instrument label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_instrument_label_dict

    def _has_instruments(self):
        """ Useful for short-circuiting circuit expansion """
        return len(self._primitive_instrument_label_dict) > 0

    def _default_primitive_prep_layer_lbl(self):
        """
        Gets the default state prep label.

        This is often used when a circuit is specified without a preparation layer.
        Returns `None` if there is no default and one *must* be specified.

        Returns
        -------
        Label or None
        """
        if len(self._primitive_prep_label_dict) == 1:
            return next(iter(self._primitive_prep_label_dict.keys()))
        else:
            return None

    def _default_primitive_povm_layer_lbl(self, sslbls):
        """
        Gets the default POVM label.

        This is often used when a circuit  is specified without an ending POVM layer.
        Returns `None` if there is no default and one *must* be specified.

        Parameters
        ----------
        sslbls : tuple or None
            The state space labels being measured, and for which a default POVM is desired.

        Returns
        -------
        Label or None
        """
        if len(self._primitive_povm_label_dict) == 1 and \
           (sslbls is None or sslbls == ('*',) or (self.state_space.num_tensor_product_blocks == 1
                                                   and self.state_space.tensor_product_block_labels(0) == sslbls)):
            return next(iter(self._primitive_povm_label_dict.keys()))
        else:
            return None

    def _has_primitive_preps(self):
        """
        Whether this model contains any state preparations.

        Returns
        -------
        bool
        """
        return len(self._primitive_prep_label_dict) > 0

    def _has_primitive_povms(self):
        """
        Whether this model contains any POVMs (measurements).

        Returns
        -------
        bool
        """
        return len(self._primitive_povm_label_dict) > 0

    def _effect_labels_for_povm(self, povm_lbl):
        """
        Gets the effect labels corresponding to the possible outcomes of POVM label `povm_lbl`.

        Parameters
        ----------
        povm_lbl : Label
            POVM label.

        Returns
        -------
        list
            A list of strings which label the POVM outcomes.
        """
        raise NotImplementedError("Derived classes must implement this!")

    def _member_labels_for_instrument(self, inst_lbl):
        """
        Get the member labels corresponding to the possible outcomes of the instrument labeled by `inst_lbl`.

        Parameters
        ----------
        inst_lbl : Label
            Instrument label.

        Returns
        -------
        list
            A list of strings which label the instrument members.
        """
        raise NotImplementedError("Derived classes must implement this!")

    # END operation container interface functions

    def circuit_layer_operator(self, layerlbl, typ="auto"):
        """
        Construct or retrieve the operation associated with a circuit layer.

        Parameters
        ----------
        layerlbl : Label
            The circuit-layer label to construct an operation for.

        typ : {'op','prep','povm','auto'}
            The type of layer `layerlbl` refers to: `'prep'` is for state
            preparation (only at the beginning of a circuit), `'povm'` is for
            a measurement: a POVM or effect label (only at the end of a circuit),
            and `'op'` is for all other "middle" circuit layers.

        Returns
        -------
        LinearOperator or State or POVM
        """
        self._clean_paramvec()
        return self._circuit_layer_operator(layerlbl, typ)

    def _circuit_layer_operator(self, layerlbl, typ):
        # doesn't call _clean_paramvec for performance
        fns = {'op': self._layer_rules.operation_layer_operator,
               'prep': self._layer_rules.prep_layer_operator,
               'povm': self._layer_rules.povm_layer_operator}
        if typ == 'auto':
            for fn in fns.values():
                try:
                    return fn(self, layerlbl, self._opcaches)
                except KeyError: pass  # Indicates failure to create op: try next type
            #raise ValueError(f"Cannot create operator for non-primitive circuit layer: {layerlbl}")
            raise ValueError("Cannot create operator for non-primitive circuit layer: %s" % str(layerlbl))
        else:
            return fns[typ](self, layerlbl, self._opcaches)

    def circuit_operator(self, circuit):
        """
        Construct or retrieve the operation associated with a circuit.

        Parameters
        ----------
        circuit : Circuit
            The circuit to construct an operation for.  This circuit should *not*
            contain any state preparation or measurement layers.

        Returns
        -------
        LinearOperator
        """
        return self.circuit_layer_operator(circuit.to_label(), typ='op')

    def _reinit_opcaches(self):
        """Called when parameter vector structure changes and self._opcaches should be cleared & re-initialized"""
        self._opcaches.clear()

    def probabilities(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        return self.sim.probs(circuit, outcomes, time)

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        resource_alloc = _ResourceAllocation(comm, mem_limit)
        return self.sim.bulk_probs(circuits, clip_to, resource_alloc, smartc)

    def _init_copy(self, copy_into, memo):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """
        self._clean_paramvec()  # make sure _paramvec is valid before copying (necessary?)
        copy_into._need_to_rebuild = True  # copy will have all gpindices = None, etc.
        copy_into._opcaches = {}  # don't copy opcaches
        super(OpModel, self)._init_copy(copy_into, memo)

    def _post_copy(self, copy_into, memo):
        """
        Called after all other copying is done, to perform "linking" between
        the new model (`copy_into`) and its members.
        """
        copy_into._sim.model = copy_into  # set copy's `.model` link
        copy_into._reinit_opcaches()
        super(OpModel, self)._post_copy(copy_into, memo)

    def copy(self):
        """
        Copy this model.

        Returns
        -------
        Model
            a (deep) copy of this model.
        """
        self._clean_paramvec()  # ensure _paramvec is rebuilt if needed
        if OpModel._pcheck: self._check_paramvec()
        ret = Model.copy(self)
        if OpModel._pcheck: ret._check_paramvec()
        return ret

    def create_modelmember_graph(self):
        """
        Generate a ModelMemberGraph for the model.

        Returns
        -------
        ModelMemberGraph
            A directed graph capturing dependencies among model members
        """
        raise NotImplementedError("Derived classes must implement this")

    def print_modelmembers(self):
        """
        Print a summary of all the members within this model.
        """
        mmg = self.create_modelmember_graph()
        mmg.print_graph()

    def is_similar(self, other_model, rtol=1e-5, atol=1e-8):
        """Whether or not two Models have the same structure.

        If `True`, then the two models are the same except for, perhaps, being
        at different parameter-space points (i.e. having different parameter vectors).
        Similar models, A and B, can be made equivalent (see :method:`is_equivalent`) by
        calling `modelA.from_vector(modelB.to_vector())`.

        Parameters
        ----------
        other_model: Model
            The model to compare against

        rtol : float, optional
            Relative tolerance used to check if floating point values are "equal", as passed to
            `numpy.allclose`.

        atol: float, optional
            Absolute tolerance used to check if floating point values are "equal", as passed to
            `numpy.allclose`.

        Returns
        -------
        bool
        """
        mmg = self.create_modelmember_graph()
        other_mmg = other_model.create_modelmember_graph()
        return mmg.is_similar(other_mmg, rtol, atol)

    def is_equivalent(self, other_model, rtol=1e-5, atol=1e-8):
        """Whether or not two Models are equivalent to each other.

        If `True`, then the two models have the same structure *and* the same
        parameters, so they are in all ways alike and will compute the same probabilities.

        Parameters
        ----------
        other_model: Model
            The model to compare against

        rtol : float, optional
            Relative tolerance used to check if floating point (including parameter) values
            are "equal", as passed to `numpy.allclose`.

        atol: float, optional
            Absolute tolerance used to check if floating point (including parameter) values
            are "equal", as passed to `numpy.allclose`.

        Returns
        -------
        bool
        """
        mmg = self.create_modelmember_graph()
        other_mmg = other_model.create_modelmember_graph()
        return mmg.is_equivalent(other_mmg, rtol, atol)

    def _format_gauge_action_matrix(self, mx, op, reduce_to_model_space, row_basis, op_gauge_basis,
                                    create_complete_basis_fn):

        from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis
        from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
        from pygsti.baseobjs.errorgenspace import ErrorgenSpace as _ErrorgenSpace
        import scipy.sparse as _sps

        #Next:
        # - make linear combos of basis els so that all (nonzero) disallowed rows become zero, i.e.,
        #    find nullspace of the matrix formed from the (nonzero) disallowed rows of mx.
        # - promote op_gauge_basis => op_gauge_space (linear combos of the same elementary basis - op_gauge_basis)
        all_sslbls = self.state_space.tensor_product_block_labels(0)

        if reduce_to_model_space:
            allowed_lbls = op.errorgen_coefficient_labels()
            allowed_lbls_set = set(allowed_lbls)
            allowed_row_basis = _ExplicitElementaryErrorgenBasis(self.state_space, allowed_lbls, basis1q=None)
            disallowed_indices = [i for i, lbl in enumerate(row_basis.labels) if lbl not in allowed_lbls_set]

            if len(disallowed_indices) > 0:
                disallowed_rows = mx[disallowed_indices, :]  # a sparse (lil) matrix
                allowed_gauge_linear_combos = _mt.nice_nullspace(disallowed_rows.toarray(), tol=1e-4)  # DENSE for now
                mx = _sps.csr_matrix(mx.dot(allowed_gauge_linear_combos))  # dot sometimes/always returns dense array
                op_gauge_space = _ErrorgenSpace(allowed_gauge_linear_combos, op_gauge_basis)  # DENSE mxs in eg-spaces
                #FOGI DEBUG: print("DEBUG => mx reduced to ", mx.shape)
            else:
                op_gauge_space = _ErrorgenSpace(_np.identity(len(op_gauge_basis), 'd'), op_gauge_basis)
        else:
            allowed_row_basis = create_complete_basis_fn(all_sslbls)
            op_gauge_space = _ErrorgenSpace(_np.identity(len(op_gauge_basis), 'd'),
                                            op_gauge_basis)
            # Note: above, do we need to store identity? could we just use the basis as a space here? or 'None'?

        # "reshape" mx so rows correspond to allowed_row_basis (the op's allowed labels)
        # (maybe make this into a subroutine?)
        assert(_sps.isspmatrix_csr(mx))
        data = []; col_indices = []; rowptr = [0]  # build up a CSR matrix manually
        allowed_lbls_set = set(allowed_row_basis.labels)
        allowed_row_indices = [(i, allowed_row_basis.label_index(lbl))
                               for i, lbl in enumerate(row_basis.labels) if lbl in allowed_lbls_set]

        for i, new_i in sorted(allowed_row_indices, key=lambda x: x[1]):
            # transfer i-th row of mx (whose rose are in row_basis) to new_i-th row of new mx

            # - first increment rowptr as needed
            while len(rowptr) <= new_i:
                rowptr.append(len(data))

            # - then add data
            col_indices.extend(mx.indices[mx.indptr[i]:mx.indptr[i + 1]])
            data.extend(mx.data[mx.indptr[i]:mx.indptr[i + 1]])
            rowptr.append(len(data))
        while len(rowptr) <= len(allowed_row_basis):  # fill in rowptr for any (empty) ending rows
            rowptr.append(len(data))
        allowed_rowspace_mx = _sps.csr_matrix((data, col_indices, rowptr),
                                              shape=(len(allowed_row_basis), mx.shape[1]), dtype=mx.dtype)

        return allowed_rowspace_mx, allowed_row_basis, op_gauge_space

    def fogi_errorgen_component_labels(self, include_fogv=False, typ='normal'):
        labels = self.fogi_store.fogi_errorgen_direction_labels(typ)
        if include_fogv:
            labels += self.fogi_store.fogv_errorgen_direction_labels(typ)
        return labels

    def fogi_errorgen_components_array(self, include_fogv=False, normalized_elem_gens=True):
        op_coeffs = self.errorgen_coefficients(normalized_elem_gens)

        if include_fogv:
            fogi_coeffs, fogv_coeffs = self.fogi_store.opcoeffs_to_fogiv_components_array(op_coeffs)
            return _np.concatenate((fogi_coeffs, fogv_coeffs))
        else:
            return self.fogi_store.opcoeffs_to_fogi_components_array(op_coeffs)

    def set_fogi_errorgen_components_array(self, components, include_fogv=False, normalized_elem_gens=True,
                                           truncate=False):
        fogi, fogv = self.fogi_store.num_fogi_directions, self.fogi_store.num_fogv_directions

        if include_fogv:
            n = fogi
            fogi_coeffs, fogv_coeffs = components[0:fogi], components[n: n + fogv]
            op_coeffs = self.fogi_store.fogiv_components_array_to_opcoeffs(fogi_coeffs, fogv_coeffs)
        else:
            fogi_coeffs = components[0:fogi]
            op_coeffs = self.fogi_store.fogi_components_array_to_opcoeffs(fogi_coeffs)

        if not normalized_elem_gens:
            def inv_rescale(coeffs):  # the inverse of the rescaling applied in fogi_errorgen_components_array
                d2 = _np.sqrt(self.dim); d = _np.sqrt(d2)
                return {lbl: (val * d if lbl.errorgen_type == 'H' else val) for lbl, val in coeffs.items()}
        else:
            def inv_rescale(coeffs): return coeffs

        for op_label, coeff_dict in op_coeffs.items():
            #TODO: update this conditional to something more robust (same conditiona in fogitools.py too)
            if isinstance(op_label, str) and op_label.startswith('rho'):
                self.preps[op_label].set_errorgen_coefficients(inv_rescale(coeff_dict), truncate=truncate)
            elif isinstance(op_label, str) and op_label.startswith('M'):
                self.povms[op_label].set_errorgen_coefficients(inv_rescale(coeff_dict), truncate=truncate)
            else:
                self.operations[op_label].set_errorgen_coefficients(inv_rescale(coeff_dict), truncate=truncate)

    def fogi_errorgen_vector(self, normalized_elem_gens=False):
        """
        Constructs a vector from all the error generator coefficients involved in the FOGI analysis of this model.

        Parameters
        ----------
        normalized_elem_gens : bool, optional
            Whether or not coefficients correspond to elementary error generators
            constructed from *normalized* Pauli matrices or not.

        Returns
        -------
        numpy.ndarray
        """
        d = self.errorgen_coefficients(normalized_elem_gens=normalized_elem_gens)
        errvec = _np.zeros(self.fogi_store.fogi_directions.shape[0], 'd')
        for op_lbl in self.fogi_store.primitive_op_labels:
            errdict = d[op_lbl]
            elem_errgen_lbls = self.fogi_store.elem_errorgen_labels_by_op[op_lbl]
            elem_errgen_indices = _slct.indices(self.fogi_store.op_errorgen_indices[op_lbl])
            for elemgen_lbl, i in zip(elem_errgen_lbls, elem_errgen_indices):
                errvec[i] = errdict.get(elemgen_lbl, 0.0)
        return errvec

    def _fogi_errorgen_vector_projection(self, space, normalized_elem_gens=False):
        """ A helper function that projects self.errorgen_vector onto the space spanned by the columns of `space` """
        errvec = self.fogi_errorgen_vector(normalized_elem_gens)
        Pspace = space @ _np.linalg.pinv(space)  # construct projector
        return Pspace @ errvec  # projected errvec

    # create map parameter indices <=> fogi_vector_indices (for each fogi store)
    def _create_model_parameter_to_fogi_errorgen_space_map(self):
        fogi_store = self.fogi_store
        num_elem_errgens, num_fogi_vecs = fogi_store.fogi_directions.shape
        param_to_fogi_errgen_space_mx = _np.zeros((num_elem_errgens, self.num_params), 'd')
        for op_label in fogi_store.primitive_op_labels:
            elem_errgen_lbls = fogi_store.elem_errorgen_labels_by_op[op_label]
            fogi_errgen_indices = _slct.indices(fogi_store.op_errorgen_indices[op_label])
            assert(len(fogi_errgen_indices) == len(elem_errgen_lbls))

            op = self.operations[op_label]
            coeff_index_lookup = {elem_lbl: i for i, elem_lbl in enumerate(op.errorgen_coefficient_labels())}
            coeff_indices = [coeff_index_lookup.get(elem_lbl, None) for elem_lbl in elem_errgen_lbls]

            # For our particularly simple parameterization (H+s) op parameter indices == coeff indices:
            assert(_np.allclose(op.errorgen_coefficients_array_deriv_wrt_params(), _np.identity(op.num_params))), \
                "Currently only supported for simple parameterizations where op parameter indices == coeff indices"
            op_param_indices = coeff_indices

            gpindices = _slct.indices(op.gpindices)
            mdl_param_indices = [(gpindices[i] if (i is not None) else None)
                                 for i in op_param_indices]
            for i_errgen, i_param in zip(fogi_errgen_indices, mdl_param_indices):
                if i_param is not None:
                    param_to_fogi_errgen_space_mx[i_errgen, i_param] = 1.0
        return param_to_fogi_errgen_space_mx

    def fogi_contribution(self, op_label, error_type='H', intrinsic_or_relational='intrinsic',
                          target='all', hessian_for_errorbars=None):
        """
        Computes a contribution to the FOGI error on a single gate.

        This method is used when partitioning the (FOGI) error on a gate in
        various ways, based on the error type, whether the error is intrinsic
        or relational, and the upon the error support.

        Parameters
        ----------
        op_label : Label
            The operation to compute a contribution for.

        error_type : {'H', 'S', 'fogi_total_error', 'fogi_infidelity'}
            The type of errors to include in the partition.  `'H'` means Hamiltonian
            and `'S'` means Pauli stochastic.  There are two options for including
            *both* H and S errors: `'fogi_total_error'` adds the Hamiltonian errors
            linearly with the Pauli tochastic errors, similar to the diamond distance;
            `'fogi_infidelity'` adds the Hamiltonian errors in quadrature to the linear
            sum of Pauli stochastic errors, similar to the entanglement or average gate
            infidelity.

        intrinsic_or_relational : {"intrinsic", "relational", "all"}
            Restrict to intrinsic or relational errors (or not, using `"all"`).

        target : tuple or "all"
            A tuple of state space (qubit) labels to restrict to, e.g., `('Q0','Q1')`.
            Note that including multiple labels selects only those quantities that
            target *all* the labels. The special `"all"` value includes quantities
            on all targets (no restriction).

        hessian_for_errorbars : numpy.ndarray, optional
            If not `None`, a hessian matrix for this model (with shape `(Np, Np)`
            where `Np == self.num_params`, the number of model paramters) that is
            used to compute and return 1-sigma error bars.

        Returns
        -------
        value : float
            The value of the requested contribution.
        errorbar : float
            The 1-sigma error bar, returned *only* if `hessian_for_errorbars` is given.
        """
        if error_type in ('H', 'S'):
            space = self.fogi_store.create_fogi_aggregate_single_op_space(op_label, error_type,
                                                                          intrinsic_or_relational, target)
            return self._fogi_contribution_single_type(error_type, space, hessian_for_errorbars)

        elif error_type in ('fogi_total_error', 'fogi_infidelity'):
            Hspace = self.fogi_store.create_fogi_aggregate_single_op_space(
                op_label, 'H', intrinsic_or_relational, target)
            Sspace = self.fogi_store.create_fogi_aggregate_single_op_space(
                op_label, 'S', intrinsic_or_relational, target)
            values = self._fogi_contribution_combined_HS_types(Hspace, Sspace, hessian_for_errorbars)
            # (total, infidelity) if hessian is None otherwise (total, total_eb, infidelity, infidelity_eb)

            if error_type == 'fogi_total_error':
                return values[0] if (hessian_for_errorbars is None) else (values[0], values[1])
            else:  # error_type == 'fogi_infidelity'
                return values[1] if (hessian_for_errorbars is None) else (values[2], values[3])

        else:
            raise ValueError("Invalid error type: '%s'" % str(error_type))

    def _fogi_contribution_single_type(self, errorgen_type, space, hessian=None):
        """
        Helper function to compute fogi contribution for a single error generator type,
        where aggregation method is unambiguous.
        Note: `space` should be a fogi-errgen-space subspace.
        """
        fogi_store = self.fogi_store
        proj_errvec = self._fogi_errorgen_vector_projection(space, normalized_elem_gens=False)
        if errorgen_type == 'H':
            val = _np.linalg.norm(proj_errvec)
        elif errorgen_type == 'S':
            val = sum(proj_errvec)
        else:
            raise ValueError("Invalid `errorgen_type` '%s' - must be 'H' or 'S'!" % str(errorgen_type))

        val = _np.real_if_close(val)
        if abs(val) < 1e-10: val = 0.0

        if hessian is not None:
            if space.size == 0:  # special case
                errbar = 0.0
            else:
                T = self._create_model_parameter_to_fogi_errorgen_space_map()
                H_errgen_space = T @ hessian @ T.T

                errgen_space_to_fogi = fogi_store.fogi_directions.toarray().T
                pinv_espace_to_fogi = _np.linalg.pinv(errgen_space_to_fogi)
                H_fogi = errgen_space_to_fogi @ H_errgen_space @ pinv_espace_to_fogi

                inv_H_fogi = _np.linalg.pinv(H_fogi)  # hessian in fogi space

                #convert fogi space back to errgen_space
                inv_H_errgen_space = pinv_espace_to_fogi @ inv_H_fogi @ errgen_space_to_fogi

                Pspace = space @ _np.linalg.pinv(space)
                proj_inv_H_errgen_space = Pspace @ inv_H_errgen_space @ Pspace.T  # np.linalg.pinv(Pspace) #.T

                if errorgen_type == 'H':
                    # elements added in quadrature, val = sqrt( sum(element^2) ) = dot(proj_errvec_hat, proj_errvec)
                    proj_errvec_hat = proj_errvec / _np.linalg.norm(proj_errvec)
                    errbar = proj_errvec_hat.T @ proj_inv_H_errgen_space @ proj_errvec_hat
                elif errorgen_type == "S":
                    # elements added, val = sum(element) = dot(ones, proj_errvec)
                    ones = _np.ones((proj_inv_H_errgen_space.shape[0], 1), 'd')
                    errbar = ones.T @ proj_inv_H_errgen_space @ ones
                else:
                    raise ValueError("Invalid `errorgen_type`!")

                if abs(errbar) < 1e-10: errbar = 0.0
                errbar = _np.sqrt(float(_np.real_if_close(errbar)))

        return val if (hessian is None) else (val, errbar)

    def _fogi_contribution_combined_HS_types(self, Hspace, Sspace, hessian=None):
        """
        Helper function to compute fogi contribution for that combined multiple
        (so far only works for H+S) error generator types, where there are multiple
        aggregation methods (and all are computed).
        Note: `space` should be a fogi-errgen-space subspace.
        """
        #TODO: maybe can combine with function above?
        errvec = self.fogi_errorgen_vector(normalized_elem_gens=False)

        Hvec = self._fogi_errorgen_vector_projection(Hspace, normalized_elem_gens=False)
        Hhat = Hvec / _np.linalg.norm(Hvec)
        Svec = _np.sum(Sspace, axis=1)  # should be all 1s and zeros
        assert(all([(_np.isclose(el, 0) or _np.isclose(el, 1.0)) for el in Svec]))

        total_error_vec = Hhat + Svec  # ~ concatenate((Hhat, Svec))
        total_error_val = _np.dot(total_error_vec, errvec)

        infidelity_vec = Hvec + Svec  # ~ concatenate((Hvec, Svec))
        infidelity_val = _np.dot(infidelity_vec, errvec)

        if hessian is not None:
            T = self._create_model_parameter_to_fogi_errorgen_space_map()

            hessian_errgen_space = T @ hessian @ T.T

            errgen_space_to_fogi = self.fogi_store.fogi_directions.toarray().T
            pinv_espace_to_fogi = _np.linalg.pinv(errgen_space_to_fogi)
            H_fogi = errgen_space_to_fogi @ hessian_errgen_space @ pinv_espace_to_fogi

            inv_H_fogi = _np.linalg.pinv(H_fogi)  # hessian in fogi space

            #convert fogi space back to errgen_space
            inv_H_errgen_space = pinv_espace_to_fogi @ inv_H_fogi @ errgen_space_to_fogi

            total_error_eb = total_error_vec[None, :] @ inv_H_errgen_space @ total_error_vec[:, None]

            infidelity_eb_vec = 2 * Hvec + Svec  # ~ concatenate((2 * Hvec, Svec))
            infidelity_eb = infidelity_eb_vec[None, :] @ inv_H_errgen_space @ infidelity_eb_vec[:, None]

            if abs(total_error_eb) < 1e-10: total_error_eb = 0.0
            total_error_eb = _np.sqrt(float(_np.real_if_close(total_error_eb)))

            if abs(infidelity_eb) < 1e-10: infidelity_eb = 0.0
            infidelity_eb = _np.sqrt(float(_np.real_if_close(infidelity_eb)))

            return total_error_val, total_error_eb, infidelity_val, infidelity_eb
        else:
            return total_error_val, infidelity_val


def _default_param_bounds(num_params):
    """Construct an array to hold parameter bounds that starts with no bounds (all bounds +-inf) """
    param_bounds = _np.empty((num_params, 2), 'd')
    param_bounds[:, 0] = -_np.inf
    param_bounds[:, 1] = +_np.inf
    return param_bounds


def _param_bounds_are_nontrivial(param_bounds):
    """Checks whether a parameter-bounds array holds any actual bounds, or if all are just +-inf """
    return _np.any(param_bounds[:, 0] != -_np.inf) or _np.any(param_bounds[:, 1] != _np.inf)
