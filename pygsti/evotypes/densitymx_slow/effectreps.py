"""
POVM effect representation classes for the `densitymx_slow` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

# import functools as _functools
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import matrixtools as _mt
from pygsti import SpaceT

class EffectRep:
    """Any representation of an "effect" in the sense of a POVM."""

    def __init__(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    def probability(self, state):
        raise NotImplementedError()


class EffectRepConjugatedState(EffectRep):
    """
    A real superket representation of an "effect" in the sense of a POVM.
    Internally uses a StateRepDense object to hold the real superket.
    """

    def __init__(self, state_rep):
        self.state_rep = state_rep
        super(EffectRepConjugatedState, self).__init__(state_rep.state_space)

    def __reduce__(self):
        return (EffectRepConjugatedState, (self.state_rep,))

    def probability(self, state):
        # can assume state is a StateRep and self.state_rep is
        return _np.dot(self.state_rep.data, state.data)  # not vdot b/c *real* data

    def to_dense(self, on_space: SpaceT):
        return self.state_rep.to_dense(on_space)


class EffectRepComputational(EffectRep):

    def __init__(self, zvals, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        assert(basis.name == 'pp'), "Only Pauli-product computational effect vectors are currently supported"
        assert(state_space.num_qudits == len(zvals))
        assert(len(zvals) <= 64), "Cannot create a Computational basis rep with >64 qubits!"
        # Current storage of computational basis states converts zvals -> 64-bit integer

        base = 1
        self.zvals_int = 0
        for v in zvals:
            assert(v in (0, 1)), "zvals must contain only 0s and 1s"
            self.zvals_int += base * v
            base *= 2  # or left shift?

        self.zvals = zvals
        self.nfactors = len(zvals)  # (or nQubits)
        self.abs_elval = 1 / (_np.sqrt(2)**self.nfactors)
        self.basis = basis

        super(EffectRepComputational, self).__init__(state_space)

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.basis, self.state_space))

    def probability(self, state):
        scratch = _np.empty(self.state_space.dim, 'd')
        Edense = self.to_dense('HilbertSchmidt', scratch)
        return _np.dot(Edense, state.data)  # not vdot b/c data is *real*

    def to_dense(self, on_space: SpaceT, outvec=None):
        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx' evotype cannot produce Hilbert-space ops!")
        return _mt.zvals_int64_to_dense(self.zvals_int, self.nfactors, outvec, False, self.abs_elval)


class EffectRepTensorProduct(EffectRep):

    def __init__(self, povm_factors, effect_labels, state_space):
        #Arrays for speeding up kron product in effect reps
        max_factor_dim = max(fct.state_space.dim for fct in povm_factors)
        kron_array = _np.ascontiguousarray(
            _np.empty((len(povm_factors), max_factor_dim), 'd'))
        factordims = _np.ascontiguousarray(
            _np.array([fct.state_space.dim for fct in povm_factors], _np.int64))

        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factordims
        self.max_factor_dim = max_factor_dim  # Unused
        state_space = _StateSpace.cast(state_space)
        assert(_np.prod(factordims) == state_space.dim)
        super(EffectRepTensorProduct, self).__init__(state_space)
        self.factor_effects_have_changed()

    def to_dense(self, on_space: SpaceT, outvec=None):

        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx' evotype cannot produce Hilbert-space ops!")

        if outvec is None:
            outvec = _np.zeros(self.state_space.dim, 'd')

        N = self.state_space.dim
        nfactors = len(self.povm_factors)
        #Put last factor at end of outvec
        k = nfactors - 1  # last factor
        off = N - self.factor_dims[k]  # offset into outvec
        for i in range(self.factor_dims[k]):
            outvec[off + i] = self.kron_array[k, i]
        sz = self.factor_dims[k]

        #Repeatedly scale&copy last "sz" elements of outputvec forward
        # (as many times as there are elements in the current factor array)
        # - but multiply *in-place* the last "sz" elements.
        for k in range(nfactors - 2, -1, -1):  # for all but the last factor
            off = N - sz * self.factor_dims[k]
            endoff = N - sz

            #For all but the final element of self.kron_array[k,:],
            # mult&copy final sz elements of outvec into position
            for j in range(self.factor_dims[k] - 1):
                mult = self.kron_array[k, j]
                for i in range(sz):
                    outvec[off + i] = mult * outvec[endoff + i]
                off += sz

            #Last element: in-place mult
            #assert(off == endoff)
            mult = self.kron_array[k, self.factor_dims[k] - 1]
            for i in range(sz):
                outvec[endoff + i] *= mult
            sz *= self.factor_dims[k]

        return outvec

    def probability(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, 'd')
        Edense = self.to_dense('HilbertSchmidt', scratch)
        return _np.dot(Edense, state.data)  # not vdot b/c data is *real*

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self.factor_dims, self.effect_labels)):
            self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense('HilbertSchmidt')

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps


class EffectRepComposed(EffectRep):
    def __init__(self, op_rep, effect_rep, op_id, state_space):
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.op_id = op_id

        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.is_compatible_with(effect_rep.state_space))

        super(EffectRepComposed, self).__init__(effect_rep.state_space)

    def __reduce__(self):
        return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id, self.state_space))

    def probability(self, state):
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.probability(state)
