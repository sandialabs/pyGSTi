"""Defines the Factory class"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import collections as _collections
import numpy as _np
import warnings as _warnings

from ..tools import matrixtools as _mt

#from . import labeldicts as _ld
from . import modelmember as _gm
from . import operation as _op
from . import instrument as _instrument
from . import povm as _povm
from .label import Label as _Lbl
from ..tools import optools as _gt
from ..tools import basistools as _bt


def op_from_factories(factory_dict, lbl):
    if lbl.args:
        lbl_without_args = _Lbl(lbl.name, lbl.sslbls)
        if lbl_without_args in factory_dict:
            return factory_dict[lbl_without_args].create_simplified_op(args=lbl.args)
            # E.g. an EmbeddedOpFactory

    lbl_name = _Lbl(lbl.name)
    if lbl_name in factory_dict:
        return factory_dict[lbl_name].create_simplified_op(args=lbl.args, sslbls=lbl.sslbls)
        # E.g. an EmbeddingOpFactory

    extra = ". Maybe you forgot the args?" if not lbl.args else ""
    raise KeyError("Cannot create operator for label `%s` from factories%s" % (str(lbl), extra))


class OpFactory(_gm.ModelMember):
    """
    An OpFactory is an object that can generate "on-demand" operators
    (can be SPAM vecs, etc., as well) for a Model.  It is assigned
    certain parameter indices (it's a ModelMember), which definie the
    block of indices it may assign to its created operations.

    The central method of an OpFactory object is the `create_op` method,
    which creates an operation that is associated with a given label.
    This is very similar to a LayerLizard's function, though a
    LayerLizard has detailed knowledge and access to a Model's internals
    whereas an OpFactory is meant to create a self-contained class of
    operators (e.g. continuously parameterized gates or on-demand
    embedding).

    This class just provides a skeleton for an operation factory -
    derived classes add the actual code for creating custom objects.
    """

    def __init__(self, dim, evotype):
        """
        Creates a new OpFactory object.

        Parameters
        ----------
        dim : int
            The state-space dimension of the operation(s) this factory builds.
            (E.g. for a single qubit represented as a density matrix, `dim=4`)

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
            The evolution type of the operation(s) this factory builds.
        """
        #self._paramvec = _np.zeros(nparams, 'd')
        _gm.ModelMember.__init__(self, dim, evotype)

    def create_object(self, args=None, sslbls=None):
        """
        Create the object which implements the operation associated
        with the given `args` and `sslbls`.

        Note to developers:
        The difference beween this method and :method:`create_op` is that
        this method just creates the foundational object without needing
        to setup it's parameter indices (a technical detail which connects
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
        obj.from_vector(self.to_vector())
        return obj

    def create_simplified_op(self, args=None, sslbls=None, item_lbl=None):
        """
        Same as create_op, but returns *simplified* operations.  In addition,
        the `item_lbl` argument must be used for POVMs and Instruments, as
        these need to know which (simple) member of themselves to return
        (this machinery still needs work).

        That is, if `create_op` returns something like a POVM or an
        Instrument, this method returns a single effect or instrument-member
        operation (a single linear-operator or SPAM vector).
        """
        op = self.create_op(args, sslbls)
        if isinstance(op, _instrument.Instrument):
            return op.simplify_operations("")[item_lbl]
        elif isinstance(op, _povm.POVM):
            return op.simplify_effects("")[item_lbl]
        else:
            return op

    def transform(self, S):
        """
        Update OpFactory so that created ops G are additionally transformed
        as inv(S) * G * S.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix
            (and it's inverse) used in the above similarity transform.
        """
        raise NotImplementedError("Cannot currently transform factories!")
        # It think we'd need to keep track of all the transform calls
        # that have been made, storing the "current S" element, and then
        # apply obj.transorm(S) within create_op(...) after creating the
        # object to return.
        #self.dirty = True

    def __str__(self):
        s = "%s object with dimension %d and %d params" % (
            self.__class__.__name__, self.dim, self.num_params())
        return s

    #Note: to_vector, from_vector, and num_params are inherited from
    # ModelMember and assume there are no parameters.


class EmbeddedOpFactory(OpFactory):
    """
    A factory that embeds a given factory's action into a single, pre-defined
    set of target sectors.
    """

    def __init__(self, stateSpaceLabels, targetLabels, factory_to_embed, dense=False):
        """
        Create a new EmbeddedOpFactory object.

        Parameters
        ----------
        stateSpaceLabels : StateSpaceLabels or a list of tuples
            This argument specifies the density matrix space upon which the
            operations this factory builds act.  If a list of tuples, each tuple
            corresponds to a block of a density matrix in the standard basis
            (and therefore a component of the direct-sum density matrix
            space). Elements of a tuple are user-defined labels beginning with
            "L" (single Level) or "Q" (two-level; Qubit) which interpret the
            d-dimensional state space corresponding to a d x d block as a tensor
            product between qubit and single level systems.  (E.g. a 2-qubit
            space might be labelled `[('Q0','Q1')]`).

        targetLabels : list of strs
            The labels contained in `stateSpaceLabels` which demarcate the
            portions of the state space acted on by the operations produced
            by `factory_to_embed` (the "contained" factory).

        factory_to_embed : OpFactory
            The factory object that is to be contained within this factory,
            and that specifies the only non-trivial action of the operations
            this factory produces.
        """
        from .labeldicts import StateSpaceLabels as _StateSpaceLabels
        self.embedded_factory = factory_to_embed
        self.state_space_labels = _StateSpaceLabels(stateSpaceLabels,
                                                    evotype=factory_to_embed._evotype)
        self.targetLabels = targetLabels
        self.dense = dense
        super(EmbeddedOpFactory, self).__init__(self.state_space_labels.dim, factory_to_embed._evotype)

        #FUTURE: somehow do all the difficult embedded op computation once at construction so we
        # don't need to keep reconstructing an Embedded op in each create_op call.
        #Embedded = _op.EmbeddedDenseOp if dense else _op.EmbeddedOp
        #dummyOp = _op.ComposedOp([], dim=factory_to_embed.dim, evotype=factor_to_embed._evotype)
        #self.embedded_op = Embedded(stateSpaceLabels, targetLabels, dummyOp)

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
        assert(sslbls is None), ("EmbeddedOpFactory objects should not be asked to create "
                                 "operations with given `sslbls` (these are already fixed!)")

        Embedded = _op.EmbeddedDenseOp if self.dense else _op.EmbeddedOp
        op = self.embedded_factory.create_op(args, sslbls)  # Note: will have its gpindices set already
        embedded_op = Embedded(self.state_space_labels, self.targetLabels, op)
        embedded_op.set_gpindices(self.gpindices, self.parent)  # Overkill, since embedded op already has indices set?
        return embedded_op

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.embedded_factory]

    def num_params(self):
        """
        Get the number of independent parameters which specify this OpFactory.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_factory.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this OpFactory.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.embedded_factory.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this OpFactory using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        Returns
        -------
        None
        """
        self.embedded_factory.from_vector(v, close, nodirty)
        if not nodirty: self.dirty = True


class EmbeddingOpFactory(OpFactory):
    """
    A factory that "on-demand" embeds a given factory or operation into any
    requested set of target sectors.  This is similar to an `EmbeddedOpFactory`
    except in this case how the "contained" operation/factory is embedded is
    *not* determined at creation time: the `sslbls` argument of
    :method:`create_op` is used instead.
    """

    def __init__(self, stateSpaceLabels, factory_or_op_to_embed, dense=False, num_target_labels=None):
        """
        Create a new EmbeddingOpFactory object.

        Parameters
        ----------
        stateSpaceLabels : StateSpaceLabels or a list of tuples
            This argument specifies the density matrix space upon which the
            operations this factory builds act.  If a list of tuples, each tuple
            corresponds to a block of a density matrix in the standard basis
            (and therefore a component of the direct-sum density matrix
            space). Elements of a tuple are user-defined labels beginning with
            "L" (single Level) or "Q" (two-level; Qubit) which interpret the
            d-dimensional state space corresponding to a d x d block as a tensor
            product between qubit and single level systems.  (E.g. a 2-qubit
            space might be labelled `[('Q0','Q1')]`).

        factory_or_op_to_embed : LinearOperator or OpFactory
            The factory or operation object that is to be contained within this
            factory.  If a linear operator, this *same* operator (not a copy)
            is embedded however is requested.  If a factory, then this object's
            `create_op` method is called with any `args` that are passed to
            the embedding-factory's `create_op` method, but the `sslbls` are
            always set to `None` (as they are processed by the embedding

        dense : bool, optional
            Whether dense embedding operations (ops which hold their entire
            "action" matrix in memory) should be created.

        num_target_labels : int, optional
            If not `None`, the number of target labels that should be expected
            (usually equal to the number of qubits the contained gate acts
            upon).  If `None`, then the length of the `sslbls` passed to this
            factory's `create_op` method is not checked at all.
        """
        from .labeldicts import StateSpaceLabels as _StateSpaceLabels
        self.embedded_factory_or_op = factory_or_op_to_embed
        self.embeds_factory = isinstance(factory_or_op_to_embed, OpFactory)
        self.state_space_labels = _StateSpaceLabels(stateSpaceLabels,
                                                    evotype=factory_or_op_to_embed._evotype)
        self.dense = dense
        self.num_target_labels = num_target_labels
        super(EmbeddingOpFactory, self).__init__(self.state_space_labels.dim, factory_or_op_to_embed._evotype)

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
        assert(sslbls is not None), ("EmbeddedOpFactory objects should be asked to create "
                                     "operations with specific `sslbls`")
        assert(self.num_target_labels is None or len(sslbls) == self.num_target_labels), \
            ("EmbeddingFactory.create_op called with the wrong number (%s) of target labels!"
             " (expected %d)") % (len(sslbls), self.num_target_labels)

        Embedded = _op.EmbeddedDenseOp if self.dense else _op.EmbeddedOp
        if self.embeds_factory:
            op = self.embedded_factory_or_op.create_op(args, None)  # Note: will have its gpindices set already
        else:
            op = self.embedded_factory_or_op
        embedded_op = Embedded(self.state_space_labels, sslbls, op)
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

    def num_params(self):
        """
        Get the number of independent parameters which specify this OpFactory.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_factory_or_op.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this OpFactory.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.embedded_factory_or_op.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this OpFactory using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        Returns
        -------
        None
        """
        self.embedded_factory_or_op.from_vector(v, close, nodirty)
        if not nodirty: self.dirty = True


class ComposedOpFactory(OpFactory):
    """
    A factory that composes a number of other factories and/or operations.

    Label arguments are passed unaltered through this factory to any component
    factories.
    """

    def __init__(self, factories_or_ops_to_compose, dim="auto", evotype="auto", dense=False):
        """
        Creates a new ComposedOpFactory.

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

        dim : int or "auto"
            Dimension of the operations produced by this factory.  Can be set
            to `"auto"` to take dimension from `factories_or_ops_to_compose[0]`
            *if* there's at least one factory or operator being composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this factory.  Can be set to `"auto"` to take
            the evolution type of `factories_or_ops_to_compose[0]` *if* there's
            at least one factory or operator being composed.

        dense : bool, optional
            Whether dense composed operations (ops which hold their entire
            "action" matrix in memory) should be created.
        """
        assert(len(factories_or_ops_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one factory/op when dim='auto'!"
        self.factors = list(factories_or_ops_to_compose)

        if dim == "auto":
            dim = factories_or_ops_to_compose[0].dim
        assert(all([dim == f.dim for f in factories_or_ops_to_compose])), \
            "All factories/ops must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = factories_or_ops_to_compose[0]._evotype
        assert(all([evotype == f._evotype for f in factories_or_ops_to_compose])), \
            "All factories/ops must have the same evolution type (%s expected)!" % evotype

        self.dense = dense
        self.is_factory = [isinstance(f, OpFactory) for f in factories_or_ops_to_compose]
        super(ComposedOpFactory, self).__init__(dim, evotype)

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
        Composed = _op.ComposedDenseOp if self.dense else _op.ComposedOp
        ops_to_compose = [f.create_op(args, sslbls) if is_f else f for is_f, f in zip(self.is_factory, self.factors)]
        op = Composed(ops_to_compose, self.dim, self._evotype)
        op.set_gpindices(self.gpindices, self.parent)  # Overkill, since composed ops already have indices set?
        return op

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors

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
        v = _np.empty(self.num_params(), 'd')
        for gate in self.factors:
            factor_local_inds = _gm._decompose_gpindices(
                self.gpindices, gate.gpindices)
            v[factor_local_inds] = gate.to_vector()
        return v

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this factory using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling from_vector"
        for gate in self.factors:
            factor_local_inds = _gm._decompose_gpindices(
                self.gpindices, gate.gpindices)
            gate.from_vector(v[factor_local_inds], close, nodirty)
        if not nodirty: self.dirty = True


#Note: to pickle these Factories we'll probably need to some work
# because they include functions.
class UnitaryOpFactory(OpFactory):
    """
    Converts a function, f(arg_tuple), that outputs a unitary matrix (operation)
    into a factory that produces a :class:`StaticDenseOp` superoperator.
    """

    def __init__(self, fn, unitary_dim, superop_basis="pp", evotype="densitymx"):
        """
        Create a new UnitaryOpFactory object.

        Parameters
        ----------
        fn : function
            A function that takes as it's only argument a tuple
            of label-arguments (arguments included in circuit labels,
            e.g. 'Gxrot;0.347') and returns a unitary matrix -- a
            complex numpy array that has dimension 2^nQubits, e.g.
            a 2x2 matrix in the 1-qubit case.

        unitary_dim : int
            The dimension of the unitary that is returned by `fn`,
            e.g. 2 for a 1-qubit factory.

        superop_basis : Basis or {"std","pp","gm","qt"}
            The basis the resulting :class:`StaticDenseOp` superoperator
            should be given in.  Usually the default of `"pp"` is what
            you want.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
            The evolution type of the operation(s) this factory builds.
        """
        self.basis = superop_basis
        self.fn = fn
        self.make_superop = bool(evotype in ("densitymx", "svterm", "cterm"))
        dim = unitary_dim**2 if self.make_superop else unitary_dim
        super(UnitaryOpFactory, self).__init__(dim, evotype)

    def create_object(self, args=None, sslbls=None):
        """
        Create the object which implements the operation associated
        with the given `args` and `sslbls`.

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
        assert(sslbls is None), "UnitaryOpFactory.create_object must be called with `sslbls=None`!"
        U = self.fn(args)
        if self.make_superop:
            superop = _bt.change_basis(_gt.unitary_to_process_mx(U), "std", self.basis)
            return _op.StaticDenseOp(superop, self._evotype)
        else:
            if self._evotype == "stabilizer":
                return _op.CliffordOp(U)
            else:
                return _op.StaticDenseOp(U, self._evotype)
