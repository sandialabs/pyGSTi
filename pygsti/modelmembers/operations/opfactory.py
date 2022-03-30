"""
Defines the OpFactory class
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

from pygsti.modelmembers.operations.staticunitaryop import StaticUnitaryOp as _StaticUnitaryOp
from pygsti.modelmembers.operations.embeddedop import EmbeddedOp as _EmbeddedOp
from pygsti.modelmembers.operations.composedop import ComposedOp as _ComposedOp

from pygsti.modelmembers import modelmember as _gm
from pygsti.modelmembers import instruments as _instrument
from pygsti.modelmembers import povms as _povm
from pygsti.baseobjs.label import Label as _Lbl
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs import basis as _basis
from pygsti.evotypes import Evotype as _Evotype
from pygsti.tools import optools as _ot


def op_from_factories(factory_dict, lbl):
    """
    Create an operator for `lbl` from the factories in `factory_dict`.

    If the label has arguments, then this function looks for an
    operator factory associated with the label without its arguments.
    If one exists, the operator is created by calling
    :method:`OpFactory.create_simplified_op`.  with the label's
    arguments.  Otherwise, it looks for a factory associated with the
    label's name (`lbl.name`) and passes both the labe's
    state-space-labels and arguments (if any) to
    :method:`OpFactory.create_simplified_op`.

    Raises a `KeyError` if a matching factory cannot be found.

    Parameters
    ----------
    factory_dict : dict
        A dictionary whose keys are labels and values are :class:`OpFactory` objects.

    lbl : Label
        The label to build an operation for.

    Returns
    -------
    LinearOperator
    """
    lbl_args = lbl.collect_args()

    lbl_without_args = lbl.strip_args() if lbl_args else lbl
    if lbl_without_args in factory_dict:
        return factory_dict[lbl_without_args].create_simplified_op(args=lbl_args)
        # E.g. an EmbeddedOpFactory or any factory labeled by a Label with sslbls

    lbl_name = _Lbl(lbl.name)
    if lbl_name in factory_dict:
        return factory_dict[lbl_name].create_simplified_op(args=lbl_args, sslbls=lbl.sslbls)
        # E.g. an EmbeddingOpFactory

    extra = ". Maybe you forgot the args?" if not lbl_args else ""
    raise KeyError("Cannot create operator for label `%s` from factories%s" % (str(lbl), extra))


class OpFactory(_gm.ModelMember):
    """
    An object that can generate "on-demand" operators (can be SPAM vecs, etc., as well) for a Model.

    It is assigned certain parameter indices (it's a ModelMember), which definie
    the block of indices it may assign to its created operations.

    The central method of an OpFactory object is the `create_op` method, which
    creates an operation that is associated with a given label.  This is very
    similar to a LayerLizard's function, though a LayerLizard has detailed
    knowledge and access to a Model's internals whereas an OpFactory is meant to
    create a self-contained class of operators (e.g. continuously parameterized
    gates or on-demand embedding).

    This class just provides a skeleton for an operation factory - derived
    classes add the actual code for creating custom objects.

    Parameters
    ----------
    state_space : StateSpace
        The state-space of the operation(s) this factory builds.

    evotype : Evotype
        The evolution type of the operation(s) this factory builds.
    """

    def __init__(self, state_space, evotype):
        #self._paramvec = _np.zeros(nparams, 'd')
        state_space = _statespace.StateSpace.cast(state_space)
        evotype = _Evotype.cast(evotype)
        _gm.ModelMember.__init__(self, state_space, evotype)

    def create_object(self, args=None, sslbls=None):
        """
        Create the object that implements the operation associated with the given `args` and `sslbls`.

        **Note to developers**
        The difference beween this method and :method:`create_op` is that
        this method just creates the foundational object without needing
        to setup its parameter indices (a technical detail which connects
        the created object with the originating factory's parameters).  The
        base-class `create_op` method calls `create_object` and then performs
        some additional setup on the returned object before returning it
        itself.  Thus, unless you have a reason for implementing `create_op`
        it's often more convenient and robust to implement this function.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, SPAMVec,
            Instrument, or POVM, depending on the label requested.
        """
        raise NotImplementedError("Derived factory classes must implement `create_object`!")

    def create_op(self, args=None, sslbls=None):
        """
        Create the operation associated with the given `args` and `sslbls`.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, SPAMVec,
            Instrument, or POVM, depending on the label requested.
        """
        obj = self.create_object(args, sslbls)  # create the object proper

        #Note: the factory's parent (usually a Model) should already
        # have allocated all of self.gpindices, so it's fine to simply
        # assign the created operation the same indices as we have.
        # (so we don't call model._init_virtual_obj as there's no need)
        obj.set_gpindices(self.gpindices, self.parent)
        obj.from_vector(self.to_vector(), dirty_value=False)
        return obj

    def create_simplified_op(self, args=None, sslbls=None, item_lbl=None):
        """
        Create the *simplified* operation associated with the given `args`, `sslbls`, and `item_lbl`.

        Similar to as :method:`create_op`, but returns a *simplified* operation
        (i.e. not a POVM or Instrument).  In addition, the `item_lbl` argument
        must be used for POVMs and Instruments, as these need to know which
        (simple) member of themselves to return (this machinery still needs
        work).

        That is, if `create_op` returns something like a POVM or an
        Instrument, this method returns a single effect or instrument-member
        operation (a single linear-operator or SPAM vector).

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        item_lbl : str, optional
            Effect or instrument-member label (index) for factories that
            create POVMs or instruments, respectively.

        Returns
        -------
        ModelMember
            Can be any type of siple operation, e.g. a LinearOperator or SPAMVec,
            depending on the label requested.
        """
        op = self.create_op(args, sslbls)
        if isinstance(op, (_instrument.Instrument, _instrument.TPInstrument)):
            return op.simplify_operations("")[item_lbl]
        elif isinstance(op, _povm.POVM):
            return op.simplify_effects("")[item_lbl]
        else:
            return op

    def transform_inplace(self, s):
        """
        Update OpFactory so that created ops `O` are additionally transformed as `inv(s) * O * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot currently transform factories!")
        # It think we'd need to keep track of all the transform_inplace calls
        # that have been made, storing the "current S" element, and then
        # apply obj.transorm(S) within create_op(...) after creating the
        # object to return.
        #self.dirty = True

    def __str__(self):
        s = "%s object with dimension %d and %d params" % (
            self.__class__.__name__, self.state_space.dim, self.num_params)
        return s

    #Note: to_vector, from_vector, and num_params are inherited from
    # ModelMember and assume there are no parameters.


class EmbeddedOpFactory(OpFactory):
    """
    A factory that embeds a given factory's action into a single, pre-defined set of target sectors.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this factory, describing the space of these that the
        operations produced by this factory act upon.

    target_labels : list of strs
        The labels contained in `state_space_labels` which demarcate the
        portions of the state space acted on by the operations produced
        by `factory_to_embed` (the "contained" factory).

    factory_to_embed : OpFactory
        The factory object that is to be contained within this factory,
        and that specifies the only non-trivial action of the operations
        this factory produces.
    """

    def __init__(self, state_space, target_labels, factory_to_embed):
        state_space = _statespace.StateSpace.cast(state_space)
        self.embedded_factory = factory_to_embed
        self.target_labels = target_labels
        super(EmbeddedOpFactory, self).__init__(state_space, factory_to_embed._evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

        #FUTURE: somehow do all the difficult embedded op computation once at construction so we
        # don't need to keep reconstructing an Embedded op in each create_op call.
        #Embedded = _op.EmbeddedDenseOp if dense else _op.EmbeddedOp
        #dummyOp = _op.ComposedOp([], dim=factory_to_embed.dim, evotype=factor_to_embed._evotype)
        #self.embedded_op = Embedded(stateSpaceLabels, target_labels, dummyOp)

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['target_labels'] = self.target_labels
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        return cls(state_space, mm_dict['target_labels'], serial_memo[mm_dict['submembers'][0]])

    def create_op(self, args=None, sslbls=None):
        """
        Create the operation associated with the given `args` and `sslbls`.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, State,
            Instrument, or POVM, depending on the label requested.
        """
        assert(sslbls is None), ("EmbeddedOpFactory objects should not be asked to create "
                                 "operations with given `sslbls` (these are already fixed!)")

        op = self.embedded_factory.create_op(args, sslbls)  # Note: will have its gpindices set already
        embedded_op = _EmbeddedOp(self.state_space, self.target_labels, op, allocated_to_parent=self.parent)
        #embedded_op.set_gpindices(self.gpindices, self.parent)  # Overkill, since embedded op already has indices set?
        # Note - adding allocated_to_parent above and commenting out set_gpindices should be fine b/c
        # 1) other factories always produce allocated ops_only_circuit, and
        # 2) when this factory is allocated (maybe assert(self.parent is not None)?), it ensures submembers are

        return embedded_op

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.embedded_factory]

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this OpFactory.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.embedded_factory.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this OpFactory.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.embedded_factory.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this OpFactory using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this factory's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        self.embedded_factory.from_vector(v, close, dirty_value)
        self.dirty = dirty_value


class EmbeddingOpFactory(OpFactory):
    """
    A factory that "on-demand" embeds a given factory or operation into any requested set of target sectors.

    This is similar to an `EmbeddedOpFactory` except in this case how the
    "contained" operation/factory is embedded is *not* determined at creation
    time: the `sslbls` argument of :method:`create_op` is used instead.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this factory, describing the space of these that the
        operations produced by this factory act upon.

    factory_or_op_to_embed : LinearOperator or OpFactory
        The factory or operation object that is to be contained within this
        factory.  If a linear operator, this *same* operator (not a copy)
        is embedded however is requested.  If a factory, then this object's
        `create_op` method is called with any `args` that are passed to
        the embedding-factory's `create_op` method, but the `sslbls` are
        always set to `None` (as they are processed by the embedding

    num_target_labels : int, optional
        If not `None`, the number of target labels that should be expected
        (usually equal to the number of qubits the contained gate acts
        upon).  If `None`, then the length of the `sslbls` passed to this
        factory's `create_op` method is not checked at all.

    allowed_sslbls_fn : callable, optional
        A boolean function that takes a single `sslbls` argument specifying the state-space
        labels for which the factory has been asked to embed `factory_or_op_to_embed`.  If
        the function returns `True` then the embedding is allowed, if `False` then an error
        is raised.
    """

    def __init__(self, state_space, factory_or_op_to_embed, num_target_labels=None, allowed_sslbls_fn=None):
        state_space = _statespace.StateSpace.cast(state_space)
        self.embedded_factory_or_op = factory_or_op_to_embed
        self.embeds_factory = isinstance(factory_or_op_to_embed, OpFactory)
        self.num_target_labels = num_target_labels
        self.allowed_sslbls_fn = allowed_sslbls_fn
        super(EmbeddingOpFactory, self).__init__(state_space, factory_or_op_to_embed._evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['num_target_labels'] = self.num_target_labels
        mm_dict['allowed_sslbls_fn'] = self.allowed_sslbls_fn.to_nice_serialization()
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        allowed_sslbls_fn = _NicelySerializable.from_nice_serialization(mm_dict['allowed_sslbls_fn'])
        return cls(state_space, serial_memo[mm_dict['submembers'][0]],
                   mm_dict['num_target_labels'], allowed_sslbls_fn)

    def create_op(self, args=None, sslbls=None):
        """
        Create the operation associated with the given `args` and `sslbls`.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, State,
            Instrument, or POVM, depending on the label requested.
        """
        assert(sslbls is not None), ("EmbeddingOpFactory objects should be asked to create "
                                     "operations with specific `sslbls`")
        assert(self.num_target_labels is None or len(sslbls) == self.num_target_labels), \
            ("EmbeddingFactory.create_op called with the wrong number (%s) of target labels!"
             " (expected %d)") % (len(sslbls), self.num_target_labels)
        if self.allowed_sslbls_fn is not None and self.allowed_sslbls_fn(sslbls) is False:
            raise ValueError("Not allowed to embed onto sslbls=" + str(sslbls))

        if self.embeds_factory:
            op = self.embedded_factory_or_op.create_op(args, sslbls)  # Note: will have its gpindices set already
        else:
            op = self.embedded_factory_or_op
        embedded_op = _EmbeddedOp(self.state_space, sslbls, op)
        embedded_op.set_gpindices(self.gpindices, self.parent)  # Overkill, since embedded op already has indices set?
        return embedded_op

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.embedded_factory_or_op]

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this OpFactory.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.embedded_factory_or_op.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this OpFactory.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.embedded_factory_or_op.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this OpFactory using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this factory's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        self.embedded_factory_or_op.from_vector(v, close, dirty_value)
        self.dirty = dirty_value


class ComposedOpFactory(OpFactory):
    """
    A factory that composes a number of other factories and/or operations.

    Label arguments are passed unaltered through this factory to any component
    factories.

    Parameters
    ----------
    factories_or_ops_to_compose : list
        List of `LinearOperator` or `OpFactory`-derived objects
        that are composed to form this factory.  There should be at least
        one factory among this list, otherwise there's no need for a
        factory.  Elements are composed with vectors in *left-to-right*
        ordering, maintaining the same convention as operation sequences
        in pyGSTi.  Note that this is *opposite* from standard matrix
        multiplication order.

    state_space : StateSpace or "auto"
        States space of the operations produced by this factory.  Can be set
        to `"auto"` to take the state space from `factories_or_ops_to_compose[0]`
        *if* there's at least one factory or operator being composed.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
        The evolution type of this factory.  Can be set to `"auto"` to take
        the evolution type of `factories_or_ops_to_compose[0]` *if* there's
        at least one factory or operator being composed.

    dense : bool, optional
        Whether dense composed operations (ops which hold their entire superoperator)
        should be created.  (Currently UNUSED - leave as default).
    """

    def __init__(self, factories_or_ops_to_compose, state_space="auto", evotype="auto", dense=False):
        assert(len(factories_or_ops_to_compose) > 0 or state_space != "auto"), \
            "Must compose at least one factory/op when state_space='auto'!"
        self.factors = list(factories_or_ops_to_compose)

        if state_space == "auto":
            state_space = factories_or_ops_to_compose[0].state_space
        assert(all([state_space.is_compatible_with(f.state_space) for f in factories_or_ops_to_compose])), \
            "All factories/ops must have compatible state spaces (%d expected)!" % str(state_space)

        if evotype == "auto":
            evotype = factories_or_ops_to_compose[0]._evotype
        assert(all([evotype == f._evotype for f in factories_or_ops_to_compose])), \
            "All factories/ops must have the same evolution type (%s expected)!" % evotype

        self.dense = dense
        self.is_factory = [isinstance(f, OpFactory) for f in factories_or_ops_to_compose]
        super(ComposedOpFactory, self).__init__(state_space, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    def create_op(self, args=None, sslbls=None):
        """
        Create the operation associated with the given `args` and `sslbls`.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, State,
            Instrument, or POVM, depending on the label requested.
        """
        ops_to_compose = [f.create_op(args, sslbls) if is_f else f for is_f, f in zip(self.is_factory, self.factors)]
        op = _ComposedOp(ops_to_compose, self.evotype, self.state_space, allocated_to_parent=self.parent)
        #op.set_gpindices(self.gpindices, self.parent)  # Overkill, since composed ops already have indices set?
        # Note - adding allocated_to_parent above and commenting out set_gpindices should be fine b/c
        # 1) other factories always produce allocated ops_only_circuit, and
        # 2) when this factory is allocated (maybe assert(self.parent is not None)?), it ensures submembers are
        return op

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this factory.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None), "Must set a ComposedOpFactory's .gpindices before calling to_vector"
        v = _np.empty(self.num_params, 'd')
        for gate in self.factors:
            factor_local_inds = _gm._decompose_gpindices(
                self.gpindices, gate.gpindices)
            v[factor_local_inds] = gate.to_vector()
        return v

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this factory using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this factory's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling from_vector"
        for gate in self.factors:
            factor_local_inds = _gm._decompose_gpindices(
                self.gpindices, gate.gpindices)
            gate.from_vector(v[factor_local_inds], close, dirty_value)
        self.dirty = dirty_value

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['dense'] = self.dense
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        factories_or_ops_to_compose = [serial_memo[i] for i in mm_dict['submembers']]
        return cls(factories_or_ops_to_compose, state_space, mm_dict['evotype'], mm_dict['dense'])


#Note: to pickle these Factories we'll probably need to some work
# because they include functions.
class UnitaryOpFactory(OpFactory):
    """
    An operation factory based on a unitary-matrix-producing function.

    Converts a function, `f(arg_tuple)`, that outputs a unitary matrix (operation)
    into a factory that produces a :class:`StaticArbitraryOp` superoperator.

    Parameters
    ----------
    fn : function
        A function that takes as it's only argument a tuple
        of label-arguments (arguments included in circuit labels,
        e.g. 'Gxrot;0.347') and returns a unitary matrix -- a
        complex numpy array that has dimension 2^nQubits, e.g.
        a 2x2 matrix in the 1-qubit case.

    state_space : StateSpace
        The state space of this factory, describing the space of these that the
        operations produced by this factory act upon.  The function `fn` should
        return a unitary matrix with dimension `state_space.udim`, e.g. a 2x2
        matrix when `state_space` describes a single qubit.

    superop_basis : Basis or {"std","pp","gm","qt"}
        The basis used to represent super-operators.  If the operations produced
        by this factor need to be given a dense superoperator representation, this
        basis is used.  Usually the default of `"pp"` is what you want.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
        The evolution type of the operation(s) this factory builds.
    """

    def __init__(self, fn, state_space, superop_basis="pp", evotype="densitymx"):
        state_space = _statespace.StateSpace.cast(state_space)
        self.basis = _basis.Basis.cast(superop_basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space

        # Compute transform matrices once and for all here, to speed up create_object calls
        std_basis = _basis.BuiltinBasis('std', state_space.dim, sparse=self.basis.sparse)
        self.transform_std_to_basis = std_basis.create_transform_matrix(self.basis)
        self.transform_basis_to_std = self.basis.create_transform_matrix(std_basis)
        self.fn = fn
        super(UnitaryOpFactory, self).__init__(state_space, evotype)

    def create_object(self, args=None, sslbls=None):
        """
        Create the object which implements the operation associated with the given `args` and `sslbls`.

        Parameters
        ----------
        args : list or tuple
            The arguments for the operation to be created.  None means no
            arguments were supplied.

        sslbls : list or tuple
            The list of state space labels the created operator should act on.
            If None, then these labels are unspecified and should be irrelevant
            to the construction of the operator (which typically, in this case,
            has some fixed dimension and no noition of state space labels).

        Returns
        -------
        ModelMember
            Can be any type of operation, e.g. a LinearOperator, State,
            Instrument, or POVM, depending on the label requested.
        """
        assert(sslbls is None), "UnitaryOpFactory.create_object must be called with `sslbls=None`!"
        U = self.fn(args)

        # Expanded call to _bt.change_basis(_ot.unitary_to_std_process_mx(U), 'std', self.basis) for speed
        std_superop = _ot.unitary_to_std_process_mx(U)
        superop_mx = _np.dot(self.transform_std_to_basis, _np.dot(std_superop, self.transform_basis_to_std))
        if self.basis.real:
            assert(_np.linalg.norm(superop_mx.imag) < 1e-8)
            superop_mx = superop_mx.real
        return _StaticUnitaryOp.quick_init(U, superop_mx, self.basis, self.evotype, self.state_space)

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['unitary_function'] = self.fn.to_nice_serialization()
        mm_dict['superop_basis'] = self.basis if isinstance(self.basis, str) \
            else self.basis.to_nice_serialization()
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        from pygsti.baseobjs.unitarygatefunction import UnitaryGateFunction as _UnitaryGateFunction
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        superop_basis = mm_dict['superop_basis']
        if isinstance(superop_basis, dict):
            superop_basis = _basis.Basis.from_nice_serialization(superop_basis)
        fn = _UnitaryGateFunction.from_nice_serialization(mm_dict['unitary_function'])
        return cls(fn, state_space, superop_basis, mm_dict['evotype'])
