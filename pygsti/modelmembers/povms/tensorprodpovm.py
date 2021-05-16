"""
Defines the TensorProductPOVM class
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
import collections as _collections
import itertools as _itertools

from .. import modelmember as _mm
from .povm import POVM as _POVM
from .tensorprodeffect import TensorProductPOVMEffect as _TensorProductPOVMEffect


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
    """

    def __init__(self, factor_povms, evotype="auto"):
        dim = _np.product([povm.dim for povm in factor_povms])

        # self.factorPOVMs
        #  Copy each POVM and set it's parent and gpindices.
        #  Assume each one's parameters are independent.
        self.factorPOVMs = [povm.copy() for povm in factor_povms]

        off = 0
        for povm in self.factorPOVMs:
            N = povm.num_params
            povm.set_gpindices(slice(off, off + N), self); off += N

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

        super(TensorProductPOVM, self).__init__(dim, evotype, items)

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
        return _np.product([len(fk) for fk in self._factor_keys])

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
            effect = _TensorProductPOVMEffect(self.factorPOVMs, elbls)
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

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
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
        OrderedDict of SPAMVecs
        """
        #Note: calling from_vector(...) on the simplified effect vectors (in
        # order) - e.g. within the finite differencing in MapForwardSimulator -  must
        # be able to properly initialize them, so need to set gpindices
        # appropriately.

        #Create a "simplified" (Model-referencing) set of factor POVMs
        factorPOVMs_simplified = []
        for p in self.factorPOVMs:
            povm = p.copy()
            povm.set_gpindices(_mm._compose_gpindices(self.gpindices,
                                                      p.gpindices), self.parent)
            factorPOVMs_simplified.append(povm)

        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        # Currently simplify *all* the effects, creating those that haven't been yet (lazy creation)
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, _TensorProductPOVMEffect(factorPOVMs_simplified, self[k].effectLbls))
             for k in self.keys()])
        return simplified

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        vl = _np.empty(self.num_params, dtype=object)
        for povm in self.factorPOVMs:
            vl[povm.gpindices] = povm.parameter_labels
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
        for povm in self.factorPOVMs:
            v[povm.gpindices] = povm.to_vector()
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
        for povm in self.factorPOVMs:
            povm.from_vector(v[povm.gpindices], close, dirty_value)

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
        #s += _mt.mx_to_string( _np.concatenate( [effect.todense() for effect in self.values()],
        #                                   axis=1), width=6, prec=2)
        return s
