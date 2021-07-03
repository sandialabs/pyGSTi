"""
Defines the ComputationalBasisPOVM class
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
import itertools as _itertools

from pygsti.modelmembers.povms.computationaleffect import ComputationalBasisPOVMEffect as _ComputationalBasisPOVMEffect
from pygsti.modelmembers.povms.povm import POVM as _POVM
from pygsti.baseobjs import statespace as _statespace


class ComputationalBasisPOVM(_POVM):
    """
    A POVM that "measures" states in the computational "Z" basis.

    Parameters
    ----------
    nqubits : int
        The number of qubits

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    qubit_filter : list, optional
        An optional list of integers specifying a subset
        of the qubits to be measured.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    @classmethod
    def from_pure_vectors(cls, pure_vectors, evotype, state_space):
        # Check if `pure_vectors` happens to be a Z-basis POVM on n-qubits
        assert(len(pure_vectors) > 0)
        if not isinstance(pure_vectors, dict):
            pure_vectors = _collections.OrderedDict(pure_vectors)
        nqubits = len(next(iter(pure_vectors.values())))

        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1) - eigenstates of sigma_z
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])  # FUTURE: make this more efficient
            lbl = ''.join(map(str, zvals))
            if not _np.allclose(testvec, pure_vectors[lbl]):
                raise ValueError("`pure_vectors` doesn't look like a Z-basis computational POVM")
        return cls(nqubits, evotype, None, state_space)

    def __init__(self, nqubits, evotype="default", qubit_filter=None, state_space=None):
        if qubit_filter is not None:
            raise NotImplementedError("Still need to implement qubit_filter functionality")

        self.nqubits = nqubits
        self.qubit_filter = qubit_filter

        #LATER - do something with qubit_filter here
        # qubits = self.qubit_filter if (self.qubit_filter is not None) else list(range(self.nqubits))

        items = []  # init as empty (lazy creation of members)
        if state_space is None:
            state_space = _statespace.QubitSpace(nqubits)
        assert(state_space.num_qubits == nqubits), "`state_space` must describe %d qubits!" % nqubits
        super(ComputationalBasisPOVM, self).__init__(state_space, evotype, items)

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        fkeys = ('0', '1')
        return bool(len(key) == self.nqubits
                    and all([(letter in fkeys) for letter in key]))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return 2**self.nqubits

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        # TODO: CHP short circuit
        if self._evotype == 'chp':
            return
            yield

        iterover = [('0', '1')] * self.nqubits
        for k in _itertools.product(*iterover):
            yield "".join(k)

    def values(self):
        """
        An iterator over the effect vectors of this POVM.
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
            # decompose key into separate factor-effect labels
            outcomes = [(0 if letter == '0' else 1) for letter in key]
            effect = _ComputationalBasisPOVMEffect(outcomes, 'pp', self._evotype, self.state_space)
            effect.set_gpindices(slice(0, 0, None), self.parent)  # computational vecs have no params
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this StabilizerZPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (ComputationalBasisPOVM, (self.nqubits, self._evotype, self.qubit_filter),
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
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if prefix: prefix += "_"
        simplified = _collections.OrderedDict(
            [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    def __str__(self):
        s = "Computational(Z)-basis POVM on %d qubits and filter %s\n" \
            % (self.nqubits, str(self.qubit_filter))
        return s
