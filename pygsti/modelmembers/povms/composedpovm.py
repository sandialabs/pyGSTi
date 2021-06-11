"""
Defines the ComposedPOVM class
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

from .composedeffect import ComposedPOVMEffect as _ComposedPOVMEffect
from .computationalpovm import ComputationalBasisPOVM as _ComputationalBasisPOVM
from .povm import POVM as _POVM
from .. import modelmember as _mm
from .. import operations as _op


class ComposedPOVM(_POVM):
    """
    TODO: update docstring
    A POVM that is effectively a *single* Lindblad-parameterized gate followed by a computational-basis POVM.

    Parameters
    ----------
    errormap : MapOperator
        The error generator action and parameterization, encapsulated in
        a gate object.  Usually a :class:`LindbladOp`
        or :class:`ComposedOp` object.  (This argument is *not* copied,
        to allow ComposedPOVMEffects to share error generator
        parameters with other gates and spam vectors.)

    povm : POVM, optional
        A sub-POVM which supplies the set of "reference" effect vectors
        that `errormap` acts on to produce the final effect vectors of
        this LindbladPOVM.  This POVM must be *static*
        (have zero parameters) and its evolution type must match that of
        `errormap`.  If None, then a :class:`ComputationalBasisPOVM` is
        used on the number of qubits appropriate to `errormap`'s dimension.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for this spam vector. Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt) (or a custom
        basis object).  If None, then this is extracted (if possible) from
        `errormap`.
    """

    def __init__(self, errormap, povm=None, mx_basis=None):
        """
        Creates a new LindbladPOVM object.

        Parameters
        ----------
        errormap : MapOperator
            The error generator action and parameterization, encapsulated in
            a gate object.  Usually a :class:`LindbladOp`
            or :class:`ComposedOp` object.  (This argument is *not* copied,
            to allow ComposedPOVMEffects to share error generator
            parameters with other gates and spam vectors.)

        povm : POVM, optional
            A sub-POVM which supplies the set of "reference" effect vectors
            that `errormap` acts on to produce the final effect vectors of
            this LindbladPOVM.  This POVM must be *static*
            (have zero parameters) and its evolution type must match that of
            `errormap`.  If None, then a :class:`ComputationalBasisPOVM` is
            used on the number of qubits appropriate to `errormap`'s dimension.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this spam vector. Allowed values are Matrix-unit (std),
            Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt) (or a custom
            basis object).  If None, then this is extracted (if possible) from
            `errormap`.
        """
        self.error_map = errormap
        state_space = self.error_map.state_space

        if mx_basis is None:
            if isinstance(errormap, _op.LindbladOp):
                mx_basis = errormap.errorgen.matrix_basis
            else:
                raise ValueError("Cannot extract a matrix-basis from `errormap` (type %s)"
                                 % str(type(errormap)))

        self.matrix_basis = mx_basis
        evotype = self.error_map._evotype

        if povm is None:
            assert(state_space.num_qubits >= 0), \
                ("A default computational-basis POVM can only be used with an"
                 " integral number of qubits!")
            povm = _ComputationalBasisPOVM(state_space.num_qubits, evotype)
        else:
            assert(povm.evotype == evotype), \
                ("Evolution type of `povm` (%s) must match that of "
                 "`errormap` (%s)!") % (povm.evotype, evotype)
            assert(povm.num_params == 0), \
                "Given `povm` must be static (have 0 parameters)!"
        self.base_povm = povm

        items = []  # init as empty (lazy creation of members)
        _POVM.__init__(self, state_space, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        return bool(key in self.base_povm)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self.base_povm)

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in self.base_povm.keys():
            yield k

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            pureVec = self.base_povm[key]
            effect = _ComposedPOVMEffect(pureVec, self.error_map)
            effect.set_gpindices(self.error_map.gpindices, self.parent)
            # initialize gpindices of "child" effect (should be in simplify_effects?)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this LindbladPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (ComposedPOVM, (self.error_map.copy(), self.base_povm.copy(), self.matrix_basis),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    def allocate_gpindices(self, starting_index, parent, memo=None):
        """
        Sets gpindices array for this object or any objects it contains (i.e. depends upon).

        Indices may be obtained from contained objects which have already been
        initialized (e.g. if a contained object is shared with other top-level
        objects), or given new indices starting with `starting_index`.

        Parameters
        ----------
        starting_index : int
            The starting index for un-allocated parameters.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : set, optional
            Used to prevent duplicate calls and self-referencing loops.  If
            `memo` contains an object's id (`id(self)`) then this routine
            will exit immediately.

        Returns
        -------
        num_new : int
            The number of *new* allocated parameters (so
            the parent should mark as allocated parameter
            indices `starting_index` to `starting_index + new_new`).
        """
        if memo is None: memo = set()
        if id(self) in memo: return 0
        memo.add(id(self))

        assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
        num_new_params = self.error_map.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
        _mm.ModelMember.set_gpindices(
            self, self.error_map.gpindices, parent)
        return num_new_params

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.error_map]

    def relink_parent(self, parent):  # Unnecessary?
        """
        Sets the parent of this object *without* altering its gpindices.

        In addition to setting the parent of this object, this method
        sets the parent of any objects this object contains (i.e.
        depends upon) - much like allocate_gpindices.  To ensure a valid
        parent is not overwritten, the existing parent *must be None*
        prior to this call.

        Parameters
        ----------
        parent : Model or ModelMember
            The parent of this POVM.

        Returns
        -------
        None
        """
        self.error_map.relink_parent(parent)
        _mm.ModelMember.relink_parent(self, parent)

    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        memo : dict, optional
            A memo dict used to avoid circular references.

        Returns
        -------
        None
        """
        if memo is None: memo = set()
        elif id(self) in memo: return
        memo.add(id(self))

        assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
        self.error_map.set_gpindices(gpindices, parent, memo)
        self.terms = {}  # clear terms cache since param indices have changed now
        _mm.ModelMember._set_only_my_gpindices(self, gpindices, parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect POVMEffects that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of POVMEffects
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.error_map.parameter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        # Recall self.base_povm.num_params == 0
        return self.error_map.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        # Recall self.base_povm.num_params == 0
        return self.error_map.to_vector()

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
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
        # Recall self.base_povm.num_params == 0
        self.error_map.from_vector(v, close, dirty_value)

    def transform_inplace(self, s):
        """
        Update each POVM effect E as s^T * E.

        Note that this is equivalent to the *transpose* of the effect vectors
        being mapped as `E^T -> E^T * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        self.error_map.spam_transform_inplace(s, 'effect')  # only do this *once*
        for lbl, effect in self.items():
            #effect._update_rep()  # these two lines mimic the bookeeping in
            effect.dirty = True   # a "effect.transform_inplace(s, 'effect')" call.
        self.dirty = True

    def __str__(self):
        s = "Lindblad-parameterized POVM of length %d\n" \
            % (len(self))
        return s
