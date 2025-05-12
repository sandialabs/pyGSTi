"""
Defines the TensorProductPOVM class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers.povms.povm import POVM as _POVM
from pygsti.modelmembers.povms.tensorprodeffect import TensorProductPOVMEffect as _TensorProductPOVMEffect
from pygsti.modelmembers import modelmember as _mm
from pygsti.baseobjs import statespace as _statespace


class TensorProductPOVM(_POVM):
    """
    A POVM that is effectively the tensor product of several other POVMs (which can be TP).

    Parameters
    ----------
    factor_povms : list of POVMs
        POVMs that will be tensor-producted together.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
        The special value `"auto"` uses the evolution type of the first
        factor *if* there are more than zero factors.

    state_space : StateSpace, optional
        The state space for this POVM.  This should be a space description
        compatible with the product of all the factors' state spaces.  If
        `None` a default compatible space will be chosen.
    """

    def __init__(self, factor_povms, evotype="auto", state_space=None):
        dim = _np.prod([povm.state_space.dim for povm in factor_povms])
        if state_space is None:
            state_space = _statespace.default_space_for_dim(dim)
        else:
            assert(state_space.dim == dim), "`state_space` is incompatible with the product of the factors' spaces!"

        self.factorPOVMs = factor_povms

        for povm in self.factorPOVMs:
            if evotype == 'auto': evotype = povm._evotype
            else: assert(evotype == povm._evotype), \
                "All factor povms must have the same evolution type"

        if evotype == 'auto':
            raise ValueError("The 'auto' evotype can only be used when there is at least one factor!")

        items = []  # init as empty (lazy creation of members)
        self._factor_keys = tuple((list(povm.keys()) for povm in factor_povms))
        self._factor_lbllens = []
        for fkeys in self._factor_keys:
            assert(len(fkeys) > 0), "Each factor POVM must have at least one effect!"
            l = len(list(fkeys)[0])  # length of the first outcome label (a string)
            assert(all([len(elbl) == l for elbl in fkeys])), \
                "All the effect labels for a given factor POVM must be the *same* length!"
            self._factor_lbllens.append(l)

        super(TensorProductPOVM, self).__init__(state_space, evotype, None, items)
        self.init_gpindices()  # initialize gpindices and subm_rpindices from sub-members

    #Note: no to_memoized_dict needed, as ModelMember version does all we need.

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        factor_povms = [serial_memo[subm_serial_id] for subm_serial_id in mm_dict['submembers']]
        return cls(factor_povms, mm_dict['evotype'], state_space)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factorPOVMs

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        i = 0
        for fkeys, lbllen in zip(self._factor_keys, self._factor_lbllens):
            if key[i:i + lbllen] not in fkeys: return False
            i += lbllen
        return True

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return _np.prod([len(fk) for fk in self._factor_keys])

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in _itertools.product(*self._factor_keys):
            yield "".join(k)

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
            elbls = []; i = 0  # decompose key into separate factor-effect labels
            for fkeys, lbllen in zip(self._factor_keys, self._factor_lbllens):
                elbls.append(key[i:i + lbllen]); i += lbllen
            # infers parent & gpindices from factor_povms
            effect = _TensorProductPOVMEffect(self.factorPOVMs, elbls, self.state_space)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this TensorProdPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (TensorProductPOVM, ([povm.copy() for povm in self.factorPOVMs],),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

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
        # Currently simplify *all* the effects, creating those that haven't been yet (lazy creation)
        if prefix: prefix += "_"
        return _collections.OrderedDict([(prefix + k, self[k]) for k in self.keys()])

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.num_params, dtype=object)
        for povm, povm_local_inds in zip(self.factorPOVMs, self._submember_rpindices):
            vl[povm_local_inds] = povm.parameter_labels
        return vl

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return sum([povm.num_params for povm in self.factorPOVMs])

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params, 'd')
        for povm, povm_local_inds in zip(self.factorPOVMs, self._submember_rpindices):
            v[povm_local_inds] = povm.to_vector()
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
        for povm, povm_local_inds in zip(self.factorPOVMs, self._submember_rpindices):
            povm.from_vector(v[povm_local_inds], close, dirty_value)

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
        for povm in self.factorPOVMs:
            povm.depolarize(amount)

        #No need to re-init effect vectors since they don't store a (dense)
        # version of their vector - they just create it from factor_povms on demand
        self.dirty = True

    def __str__(self):
        s = "Tensor-product POVM with %d factor POVMs\n" % len(self.factorPOVMs)
        #s += " and final effect labels " + ", ".join(self.keys()) + "\n"
        for i, povm in enumerate(self.factorPOVMs):
            s += "Factor %d: " % i
            s += str(povm)

        #s = "Tensor-product POVM with effect labels:\n"
        #s += ", ".join(self.keys()) + "\n"
        #s += " Effects (one per column):\n"
        #s += _mt.mx_to_string( _np.concatenate( [effect.toarray() for effect in self.values()],
        #                                   axis=1), width=6, prec=2)
        return s
