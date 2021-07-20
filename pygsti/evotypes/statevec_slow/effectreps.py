"""
POVM effect representation classes for the `statevec_slow` evolution type.
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

from pygsti.baseobjs.statespace import StateSpace as _StateSpace


class EffectRep(object):
    def __init__(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    def probability(self, state):
        return abs(self.amplitude(state))**2

    def amplitude(self, state):
        raise NotImplementedError()


class EffectRepConjugatedState(EffectRep):
    def __init__(self, state_rep):
        self.state_rep = state_rep
        super(EffectRepConjugatedState, self).__init__(state_rep.state_space)

    def __str__(self):
        return str(self.state_rep.data)

    def to_dense(self, on_space):
        return self.state_rep.to_dense(on_space)

    def amplitude(self, state):
        # can assume state is a StateRep
        return _np.vdot(self.state_rep.data, state.data)


class EffectRepComputational(EffectRep):
    def __init__(self, zvals, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_qubits == len(zvals))
        assert(len(zvals) <= 64), "Cannot create a Computational basis rep with >64 qubits!"
        # Current storage of computational basis states converts zvals -> 64-bit integer

        # Different than DM counterpart
        # as each factor only has *1* nonzero element so final state has only a
        # *single* nonzero element!  We just have to figure out where that
        # single element lies (compute it's index) based on the given zvals.

        # Assume, like tensorprod, that factor ordering == kron ordering
        # so nonzer_index = kron( factor[0], factor[1], ... factor[N-1] ).

        base = 2**(len(zvals) - 1)
        self.nonzero_index = 0
        self.zvals = zvals
        self.basis = basis
        for k, v in enumerate(zvals):
            assert(v in (0, 1)), "zvals must contain only 0s and 1s"
            self.nonzero_index += base * v
            base //= 2  # or right shift?
        super(EffectRepComputational, self).__init__(state_space)

    def to_dense(self, on_space, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if on_space not in ('minimal', 'Hilbert'):
            raise ValueError('statevec evotype cannot (yet) generate dense Hilbert-Schmidt effect vectors')
        if not trust_outvec_sparsity:
            outvec[:] = 0  # reset everything to zero
        outvec[self.nonzero_index] = 1.0
        return outvec

    def amplitude(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.to_dense('Hilbert', scratch)
        return _np.vdot(Edense, state.data)


class EffectRepTensorProduct(EffectRep):

    def __init__(self, povm_factors, effect_labels, state_space):
        #Arrays for speeding up kron product in effect reps
        max_factor_dim = max(fct.dim for fct in povm_factors)
        kron_array = _np.ascontiguousarray(
            _np.empty((len(povm_factors), max_factor_dim), complex))
        factordims = _np.ascontiguousarray(
            _np.array([fct.state_space.udim for fct in povm_factors], _np.int64))

        #dim = _np.product(factordims)
        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factordims
        self.nfactors = len(self.povm_factors)
        self.max_factor_dim = max_factor_dim  # Unused
        state_space = _StateSpace.cast(state_space)
        assert(_np.product(factordims) == state_space.udim)
        super(EffectRepTensorProduct, self).__init__(state_space)
        self.factor_effects_have_changed()

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self._fast_kron_factordims, self.effectLbls)):
            self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense('Hilbert')

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps

    def to_dense(self, on_space, scratch=None):
        #OLD & SLOW:
        #if len(self.factors) == 0: return _np.empty(0, complex)
        #factorPOVMs = self.factors
        #ret = factorPOVMs[0][self.effectLbls[0]].to_dense()
        #for i in range(1, len(factorPOVMs)):
        #    ret = _np.kron(ret, factorPOVMs[i][self.effectLbls[i]].to_dense())
        #return ret

        if scratch is None:
            scratch = _np.empty(self.udim, complex)
        outvec = scratch

        N = self.udim
        #Put last factor at end of outvec
        k = self.nfactors - 1  # last factor
        off = N - self.factor_dims[k]  # offset into outvec
        for i in range(self.factor_dims[k]):
            outvec[off + i] = self.kron_array[k, i]
        sz = self.factor_dims[k]

        #Repeatedly scale&copy last "sz" elements of outputvec forward
        # (as many times as there are elements in the current factor array)
        # - but multiply *in-place* the last "sz" elements.
        for k in range(self.nfactors - 2, -1, -1):  # for all but the last factor
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

    def amplitude(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.to_dense('Hilbert', scratch)
        return _np.vdot(Edense, state.data)


class EffectRepComposed(EffectRep):
    def __init__(self, op_rep, effect_rep, op_id, state_space):
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.op_id = op_id

        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.is_compatible_with(effect_rep.state_space))

        super(EffectRepComposed, self).__init__(effect_rep.state_space)

    #def __reduce__(self):
    #    return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id, self.state_space))

    def probability(self, state):
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.probability(state)

    def amplitude(self, state):  # allow scratch to be passed in?
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.amplitude(state)
