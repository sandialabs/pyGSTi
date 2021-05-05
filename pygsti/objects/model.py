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

import numpy as _np
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time
import uuid as _uuid
import bisect as _bisect
import copy as _copy

from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import symplectic as _symp

from . import modelmember as _gm
from . import circuit as _cir
from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import instrument as _instrument
from . import labeldicts as _ld
from . import gaugegroup as _gg
from . import forwardsim as _fwdsim
from . import matrixforwardsim as _matrixfwdsim
from . import mapforwardsim as _mapfwdsim
from . import termforwardsim as _termfwdsim
from . import explicitcalc as _explicitcalc

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from .label import Label as _Label
from .circuitlist import CircuitList as _CircuitList
from .layerrules import LayerRules as _LayerRules
from .resourceallocation import ResourceAllocation as _ResourceAllocation

MEMLIMIT_FOR_NONGAUGE_PARAMS = None


class Model(object):
    """
    A predictive model for a Quantum Information Processor (QIP).

    The main function of a `Model` object is to compute the outcome
    probabilities of :class:`Circuit` objects based on the action of the
    model's ideal operations plus (potentially) noise which makes the
    outcome probabilities deviate from the perfect ones.
v
    Parameters
    ----------
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.
    """

    def __init__(self, state_space_labels):
        """
        Creates a new Model.  Rarely used except from derived classes
        `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.
        """
        if isinstance(state_space_labels, _ld.StateSpaceLabels):
            self._state_space_labels = state_space_labels
        else:
            self._state_space_labels = _ld.StateSpaceLabels(state_space_labels)

        self._num_modeltest_params = None
        self._hyperparams = {}
        self._paramvec = _np.zeros(0, 'd')
        self._paramlbls = _np.empty(0, dtype=object)
        self.uuid = _uuid.uuid4()  # a Model's uuid is like a persistent id(), useful for hashing

    @property
    def state_space_labels(self):
        """
        State space labels

        Returns
        -------
        StateSpaceLabels
        """
        return self._state_space_labels

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
                    memForNumGaugeParams = self.num_elements * (self.num_params + self.dim**2) \
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
    def parameter_labels(self):
        """
        A list of labels, usually of the form `(op_label, string_description)` describing this model's parameters.
        """
        return self._paramlbls

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
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    basis : Basis
        The basis used for the state space by dense operator representations.

    evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
        The evolution type of this model, describing how states are
        represented, allowing compatibility checks with (super)operator
        objects.

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

    def __init__(self, state_space_labels, basis, evotype, layer_rules, simulator="auto"):
        """
        Creates a new OpModel.  Rarely used except from derived classes `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.

        basis : Basis
            The basis used for the state space by dense operator representations.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator
            objects.

        layer_rules : LayerRules
            The "layer rules" used for constructing operators for circuit
            layers.  This functionality is essential to using this model to
            simulate ciruits, and is typically supplied by derived classes.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The forward simulator this model should use.  `"auto"`
            tries to determine and instantiate the best type automatically.
        """
        self._evotype = evotype
        self._set_state_space(state_space_labels, basis)
        #sets self._state_space_labels, self._basis, self._dim

        super(OpModel, self).__init__(self.state_space_labels)  # do this as soon as possible

        self._layer_rules = layer_rules if (layer_rules is not None) else _LayerRules()
        self._opcaches = {}  # dicts of non-primitive operations (organized by derived class)
        self._need_to_rebuild = True  # whether we call _rebuild_paramvec() in to_vector() or num_params()
        self.dirty = False  # indicates when objects and _paramvec may be out of sync
        self.sim = simulator  # property setter does nontrivial initialization (do this *last*)

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
        return self._sim

    @sim.setter
    def sim(self, simulator):
        if simulator == "auto":
            d = self.dim if (self.dim is not None) else 0
            simulator = "matrix" if d <= 16 else "map"

        if simulator == "matrix":
            self._sim = _matrixfwdsim.MatrixForwardSimulator(self)
        elif simulator == "map":
            self._sim = _mapfwdsim.MapForwardSimulator(self, max_cache_size=0)  # default is to *not* use a cache
        else:
            assert(isinstance(simulator, _fwdsim.ForwardSimulator)), "`simulator` argument must be a ForwardSimulator!"
            self._sim = simulator
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
            assert(basis.dim == self.state_space_labels.dim), \
                "Cannot set basis w/dim=%d when sslbls dim=%d!" % (basis.dim, self.state_space_labels.dim)
            self._basis = basis
        else:  # create a basis with the proper structure & dimension
            self._basis = _Basis.cast(basis, self.state_space_labels)

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
        if isinstance(lbls, _ld.StateSpaceLabels):
            self._state_space_labels = lbls
        else:
            self._state_space_labels = _ld.StateSpaceLabels(lbls, evotype=self._evotype)
        self.basis = basis  # invokes basis setter to set self._basis

        #Operator dimension of this Model
        self._dim = self.state_space_labels.dim
        #e.g. 4 for 1Q (densitymx) or 2 for 1Q (statevec)

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
        return self._dim

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

            #Note: lbl args used *just* for potential debugging - could strip out once
            # we're confident this code always works.
            def clean_single_obj(obj, lbl):  # sync an object's to_vector result w/_paramvec
                if obj.dirty:
                    w = obj.to_vector()
                    chk_norm = _np.linalg.norm(self._paramvec[obj.gpindices] - w)
                    #print(lbl, " is dirty! vec = ", w, "  chk_norm = ",chk_norm)
                    if (not _np.isfinite(chk_norm)) or chk_norm > TOL:
                        self._paramvec[obj.gpindices] = w
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
                obj.from_vector(self._paramvec[obj.gpindices], dirty_value=False)
                #object is known to be consistent with _paramvec

            self.dirty = False
            self._reinit_opcaches()  # changes to parameter vector structure invalidate cached ops

        if OpModel._pcheck: self._check_paramvec()

    def _mark_for_rebuild(self, modified_obj=None):
        #re-initialze any members that also depend on the updated parameters
        self._need_to_rebuild = True
        for _, o in self._iter_parameterized_objs():
            if o._obj_refcount(modified_obj) > 0:
                o.clear_gpindices()  # ~ o.gpindices = None but works w/submembers
                # (so params for this obj will be rebuilt)
        self.dirty = True
        #since it's likely we'll set at least one of our object's .dirty flags
        # to True (and said object may have parent=None and so won't
        # auto-propagate up to set this model's dirty flag (self.dirty)

    def _print_gpindices(self):
        print("PRINTING MODEL GPINDICES!!!")
        for lbl, obj in self._iter_parameterized_objs():
            print("LABEL ", lbl)
            obj._print_gpindices()

    def _rebuild_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        v = self._paramvec; Np = len(self._paramvec)  # NOT self.num_params since the latter calls us!
        vl = self._paramlbls
        off = 0; shift = 0
        #print("DEBUG: rebuilding...")

        #Step 1: remove any unused indices from paramvec and shift accordingly
        used_gpindices = set()
        for _, obj in self._iter_parameterized_objs():
            if obj.gpindices is not None:
                assert(obj.parent is self), "Member's parent is not set correctly (%s)!" % str(obj.parent)
                used_gpindices.update(obj.gpindices_as_array())
            else:
                assert(obj.parent is self or obj.parent is None)
                #Note: ok for objects to have parent == None if their gpindices is also None

        indices_to_remove = sorted(set(range(Np)) - used_gpindices)

        if len(indices_to_remove) > 0:
            #print("DEBUG: Removing %d params:"  % len(indices_to_remove), indices_to_remove)
            v = _np.delete(v, indices_to_remove)
            vl = _np.delete(vl, indices_to_remove)
            def get_shift(j): return _bisect.bisect_left(indices_to_remove, j)
            memo = set()  # keep track of which object's gpindices have been set
            for _, obj in self._iter_parameterized_objs():
                if obj.gpindices is not None:
                    if id(obj) in memo: continue  # already processed
                    if isinstance(obj.gpindices, slice):
                        new_inds = _slct.shift(obj.gpindices,
                                               -get_shift(obj.gpindices.start))
                    else:
                        new_inds = []
                        for i in obj.gpindices:
                            new_inds.append(i - get_shift(i))
                        new_inds = _np.array(new_inds, _np.int64)
                    obj.set_gpindices(new_inds, self, memo)

        # Step 2: add parameters that don't exist yet
        #  Note that iteration order (that of _iter_parameterized_objs) determines
        #  parameter index ordering, so "normally" an object that occurs before
        #  another in the iteration order will have gpindices which are lower - and
        #  when new indices are allocated we try to maintain this normal order by
        #  inserting them at an appropriate place in the parameter vector.
        #  off : holds the current point where new params should be inserted
        #  shift : holds the amount existing parameters that are > offset (not in `memo`) should be shifted
        # Note: Adding more explicit "> offset" logic may obviate the need for the memo arg?
        memo = set()  # keep track of which object's gpindices have been set
        for lbl, obj in self._iter_parameterized_objs():

            if shift > 0 and obj.gpindices is not None:
                if isinstance(obj.gpindices, slice):
                    obj.set_gpindices(_slct.shift(obj.gpindices, shift), self, memo)
                else:
                    obj.set_gpindices(obj.gpindices + shift, self, memo)  # works for integer arrays

            if obj.gpindices is None or obj.parent is not self:
                #Assume all parameters of obj are new independent parameters
                num_new_params = obj.allocate_gpindices(off, self, memo)
                objvec = obj.to_vector()  # may include more than "new" indices
                objlbls = _np.empty(obj.num_params, dtype=object)
                objlbls[:] = [(lbl, obj_plbl) for obj_plbl in obj.parameter_labels]
                if num_new_params > 0:
                    new_local_inds = _gm._decompose_gpindices(obj.gpindices, slice(off, off + num_new_params))
                    assert(len(objvec[new_local_inds]) == num_new_params)
                    v = _np.insert(v, off, objvec[new_local_inds])
                    try:
                        vl = _np.insert(vl, off, objlbls[new_local_inds])
                    except:
                        import bpdb; bpdb.set_trace()
                        print("DB:")
                # print("objvec len = ",len(objvec), "num_new_params=",num_new_params,
                #       " gpinds=",obj.gpindices) #," loc=",new_local_inds)

                #obj.set_gpindices( slice(off, off+obj.num_params), self )
                #shift += obj.num_params
                #off += obj.num_params

                shift += num_new_params
                off += num_new_params
                #print("DEBUG: %s: alloc'd & inserted %d new params.  indices = " \
                #      % (str(lbl),obj.num_params), obj.gpindices, " off=",off)
            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds) > 0 else -1; L = len(v)
                #print("DEBUG: %s: existing indices = " % (str(lbl)), obj.gpindices, " M=",M," L=",L)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.
                    w = obj.to_vector()
                    wl = _np.empty(obj.num_params, dtype=object)
                    wl[:] = [(lbl, obj_plbl) for obj_plbl in obj.parameter_labels]
                    v = _np.concatenate((v, _np.empty(M + 1 - L, 'd')), axis=0)  # [v.resize(M+1) doesn't work]
                    vl = _np.concatenate((vl, _np.empty(M + 1 - L, dtype=object)), axis=0)
                    shift += M + 1 - L
                    for ii, i in enumerate(inds):
                        if i >= L:
                            v[i] = w[ii]
                            vl[i] = wl[ii]
                    #print("DEBUG:    --> added %d new params" % (M+1-L))
                if M >= 0:  # M == -1 signifies this object has no parameters, so we'll just leave `off` alone
                    off = M + 1

        self._paramvec = v
        self._paramlbls = vl
        #print("DEBUG: Done rebuild: %d params" % len(v))

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
            assert(obj.parent is self), "Virtual obj has incorrect parent already set!"
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
        for _, obj in self._iter_parameterized_objs():
            obj.from_vector(v[obj.gpindices], close, dirty_value=False)
            # dirty_value=False => obj.dirty = False b/c object is known to be consistent with _paramvec

        # Call from_vector on elements of the cache
        for opcache in self._opcaches.values():
            for obj in opcache.values():
                obj.from_vector(v[obj.gpindices], close, dirty_value=False)

        if OpModel._pcheck: self._check_paramvec()

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
           (sslbls is None or sslbls == ('*',) or (len(self.state_space_labels.labels) == 1
                                                   and self.state_space_labels.labels[0] == sslbls)):
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
        LinearOperator or SPAMVec
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
