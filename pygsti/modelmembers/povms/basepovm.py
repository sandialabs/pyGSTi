"""
Defines the _BasePOVM class (a base class for other POVMs, not to be used independently)
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

import numpy as _np

from pygsti.modelmembers.povms.complementeffect import ComplementPOVMEffect as _ComplementPOVMEffect
from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect
from pygsti.modelmembers.povms.fulleffect import FullPOVMEffect as _FullPOVMEffect
from pygsti.modelmembers.povms.povm import POVM as _POVM
from pygsti.modelmembers import modelmember as _mm
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


class _BasePOVM(_POVM):
    """ The base behavior for both UnconstrainedPOVM and TPPOVM """

    def __init__(self, effects, evotype=None, state_space=None, preserve_sum=False, called_from_reduce=False):
        """
        Creates a new BasePOVM object.

        Parameters
        ----------
        effects : dict of POVMEffects or array-like
            A dict (or list of key,value pairs) of the effect vectors.

        evotype : Evotype or str, optional
            The evolution type.  If `None`, the evotype is inferred
            from the first effect vector.  If `len(effects) == 0` in this case,
            an error is raised.

        state_space : StateSpace, optional
            The state space for this POVM.  If `None`, the space is inferred
            from the first effect vector.  If `len(effects) == 0` in this case,
            an error is raised.

        preserve_sum : bool, optional
            If true, the sum of `effects` is taken to be a constraint
            and so the final effect vector is made into a
            :class:`ComplementPOVMEffect`.
        """
        if isinstance(effects, dict):
            items = [(k, v) for k, v in effects.items()]  # gives definite ordering of effects
        elif isinstance(effects, list):
            items = effects  # assume effects is already an ordered (key,value) list
        else:
            raise ValueError("Invalid `effects` arg of type %s" % type(effects))

        if preserve_sum:
            assert(len(items) > 1), "Cannot create a TP-POVM with < 2 effects!"
            self.complement_label = items[-1][0]
            comp_obj = items[-1][1]  # current object & then value of complement vec
            comp_val = comp_obj.to_dense() if isinstance(comp_obj, _POVMEffect) else _np.array(comp_obj)
        else:
            self.complement_label = None

        if evotype is not None:
            evotype = _Evotype.cast(evotype)  # e.g., resolve "default"

        #Copy each effect vector and set it's parent and gpindices.
        # Assume each given effect vector's parameters are independent.
        copied_items = []
        paramlbls = []
        for k, v in items:
            if k == self.complement_label: continue
            if isinstance(v, _POVMEffect):
                effect = v if (not preserve_sum) else v.copy()  # .copy() just to de-allocate parameters
            else:
                assert(evotype is not None), "Must specify `evotype` when effect vectors are not POVMEffect objects!"
                effect = _FullPOVMEffect(v, evotype, state_space)

            if evotype is None: evotype = effect.evotype
            else: assert(evotype == effect.evotype), \
                "All effect vectors must have the same evolution type"

            if state_space is None: state_space = effect.state_space
            assert(state_space.is_compatible_with(effect.state_space)), \
                "All effect vectors must have compatible state spaces!"

            paramlbls.extend(effect.parameter_labels)
            copied_items.append((k, effect))
        items = copied_items

        if evotype is None:
            raise ValueError("Could not determine evotype - please specify `evotype` directly!")
        if state_space is None:
            raise ValueError("Could not determine state space - please specify `state_space` directly!")

        #Add a complement effect if desired
        if self.complement_label is not None:  # len(items) > 0 by assert
            non_comp_effects = [v for k, v in items]
            identity_for_complement = _np.array(sum([v.to_dense().reshape(comp_val.shape) for v in non_comp_effects])
                                                + comp_val, 'd')  # ensure shapes match before summing
            complement_effect = _ComplementPOVMEffect(
                identity_for_complement, non_comp_effects)
            items.append((self.complement_label, complement_effect))

        super(_BasePOVM, self).__init__(state_space, evotype, None, items)
        if not called_from_reduce: self.init_gpindices()  # initialize our gpindices based on sub-members
        self._paramlbls = _np.array(paramlbls, dtype=object)

    def submembers(self):
        """
        Returns a sequence of any sub-ModelMember objects contained in this one.

        Sub-members are processed by other :class:`ModelMember` methods
        (e.g. `unlink_parent` and `set_gpindices`) as though the parent
        object is *just* a container for these sub-members and has no
        parameters of its own.  Member objects that contain other members
        *and* possess their own independent parameters should implement
        the appropriate `ModelMember` functions (usually just
        `allocate_gpindices`, using the base implementation as a reference).

        Returns
        -------
        list or tuple
        """
        return tuple(self.values())  # what about complement effect?

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
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['effect_labels'] = list(self.keys())  # labels of the submember effects

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        effects = {lbl: serial_memo[subm_serial_id]
                   for lbl, subm_serial_id in zip(mm_dict['effect_labels'], mm_dict['submembers'])}
        return cls(effects, mm_dict['evotype'], state_space)  # Note: __init__ call signature of derived classes

    #def _reset_member_gpindices(self):
    #    """
    #    Sets gpindices for all non-complement items.  Assumes all non-complement
    #    vectors have *independent* parameters (for now).
    #    """
    #    Np = 0
    #    for k, effect in self.items():
    #        if k == self.complement_label: continue
    #        N = effect.num_params
    #        pslc = slice(Np, Np + N)
    #        if effect.gpindices != pslc:
    #            effect.set_gpindices(pslc, self)
    #        Np += N
    #    self.Np = Np
    #
    #def _rebuild_complement(self, identity_for_complement=None):
    #    """ Rebuild complement vector (in case other vectors have changed) """
    #
    #    if self.complement_label is not None and self.complement_label in self:
    #        non_comp_effects = [v for k, v in self.items()
    #                            if k != self.complement_label]
    #
    #        if identity_for_complement is None:
    #            identity_for_complement = self[self.complement_label].identity
    #
    #        complement_effect = _ComplementPOVMEffect(
    #            identity_for_complement, non_comp_effects)
    #        complement_effect.set_gpindices(slice(0, self.Np), self)  # all parameters
    #
    #        #Assign new complement effect without calling our __setitem__
    #        old_ro = self._readonly; self._readonly = False
    #        _POVM.__setitem__(self, self.complement_label, complement_effect)
    #        self._readonly = old_ro

    def __setitem__(self, key, value):
        if not self._readonly:  # when readonly == False, we're initializing
            return super(_BasePOVM, self).__setitem__(key, value)

        raise NotImplementedError("TODO: fix ability to set POVM effects after initialization")
        #if key == self.complement_label:
        #    raise KeyError("Cannot directly assign the complement effect vector!")
        #value = value.copy() if isinstance(value, _POVMEffect) else \
        #    _FullPOVMEffect(value)   # EVOTYPE -----------------------------------------???????????????????????????????
        #_collections.OrderedDict.__setitem__(self, key, value)
        #self._reset_member_gpindices()
        #self._rebuild_complement()

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
        if prefix: prefix = prefix + "_"
        simplified = _collections.OrderedDict()
        for lbl, effect in self.items():
            simplified[prefix + lbl] = effect

        return simplified

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params, 'd')
        for (lbl, effect), effect_local_inds in zip(self.items(), self._submember_rpindices):
            if lbl == self.complement_label: continue
            v[effect_local_inds] = effect.to_vector()
        return v

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
        for (lbl, effect), effect_local_inds in zip(self.items(), self._submember_rpindices):
            if lbl == self.complement_label: continue
            effect.from_vector(v[effect_local_inds], close, dirty_value)
        if self.complement_label:  # re-init Ec
            self[self.complement_label]._construct_vector()

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
        """
        for lbl, effect in self.items():
            if lbl == self.complement_label: continue
            effect.transform_inplace(s)

        if self.complement_label:
            #Other effects being transformed transforms the complement,
            # so just check that the transform preserves the identity.
            TOL = 1e-6
            identityVec = self[self.complement_label].identity.to_dense().reshape((-1, 1))
            SmxT = _np.transpose(s.transform_matrix)
            assert(_np.linalg.norm(identityVec - _np.dot(SmxT, identityVec)) < TOL),\
                ("Cannot transform complement effect in a way that doesn't"
                 " preserve the identity!")
            self[self.complement_label]._construct_vector()

        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this POVM by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        for lbl, effect in self.items():
            if lbl == self.complement_label:
                #Don't depolarize complements since this will depol the
                # other effects via their shared params - cleanup will update
                # any complement vectors
                continue
            effect.depolarize(amount)

        if self.complement_label:
            # depolarization of other effects "depolarizes" the complement
            self[self.complement_label]._construct_vector()
        self.dirty = True
