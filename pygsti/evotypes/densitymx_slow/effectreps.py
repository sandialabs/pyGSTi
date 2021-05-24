"""
POVM effect representation classes for the `densitymx_slow` evolution type.
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
#import functools as _functools
from .. import basereps as _basereps
from ...models.statespace import StateSpace as _StateSpace


class EffectRep(_basereps.EffectRep):
    def __init__(self, state_space):
        self.state_space = _StateSpace.cast(state_space)

    def probability(self, state):
        raise NotImplementedError()


class EffectRepConjugatedState(EffectRep):

    def __init__(self, state_rep):
        self.state_rep = state_rep
        super(EffectRepConjugatedState, self).__init__(state_rep.state_space)

    def __reduce__(self):
        return (EffectRepConjugatedState, (self.state_rep,))

    def probability(self, state):
        # can assume state is a StateRep and self.state_rep is
        return _np.dot(self.state_rep.base, state.base)  # not vdot b/c *real* data

    def to_dense(self):
        return self.state_rep.to_dense()


class EffectRepComputational(EffectRep):

    def __init__(self, zvals, state_space):
        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_qubits == len(zvals))
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

        super(EffectRepComputational, self).__init__(state_space)

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.dim, self.state_space))

    def parity(self, x):
        """recursively divide the (64-bit) integer into two equal
           halves and take their XOR until only 1 bit is left """
        x = (x & 0x00000000FFFFFFFF) ^ (x >> 32)
        x = (x & 0x000000000000FFFF) ^ (x >> 16)
        x = (x & 0x00000000000000FF) ^ (x >> 8)
        x = (x & 0x000000000000000F) ^ (x >> 4)
        x = (x & 0x0000000000000003) ^ (x >> 2)
        x = (x & 0x0000000000000001) ^ (x >> 1)
        return x & 1  # return the last bit (0 or 1)

    def to_dense(self, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if not trust_outvec_sparsity:
            outvec[:] = 0  # reset everything to zero

        N = self.nfactors

        # there are nQubits factors
        # each factor (4-element, 1Q dmvec) has 2 zero elements and 2 nonzero ones
        # loop is over all non-zero elements of the final outvec by looping over
        #  all the sets of *entirely* nonzero elements from the factors.

        # Let the two possible nonzero elements of the k-th factor be represented
        # by the k-th bit of `finds` below, which ranges from 0 to 2^nFactors-1
        for finds in range(2**N):

            #Create the final index (within outvec) corresponding to finds
            # assume, like tensorprod, that factor ordering == kron ordering
            # so outvec = kron( factor[0], factor[1], ... factor[N-1] ).
            # Let factorDim[k] == 4**(N-1-k) be the stride associated with the k-th index
            # Whenever finds[bit k] == 0 => finalIndx += 0*factorDim[k]
            #          finds[bit k] == 1 => finalIndx += 3*factorDim[k] (3 b/c factor's 2nd nonzero el is at index 3)
            finalIndx = sum([3 * (4**(N - 1 - k)) for k in range(N) if bool(finds & (1 << k))])

            #Determine the sign of this element (the element is either +/- (1/sqrt(2))^N )
            # A minus sign is picked up whenever finds[bit k] == 1 (which means we're looking
            # at the index=3 element of the factor vec) AND self.zvals_int[bit k] == 1
            # (which means it's a [1 0 0 -1] state rather than a [1 0 0 1] state).
            # Since we only care whether the number of minus signs is even or odd, we can
            # BITWISE-AND finds with self.zvals_int (giving an integer whose binary-expansion's
            # number of 1's == the number of minus signs) and compute the parity of this.
            minus_sign = self.parity(finds & self.zvals_int)

            outvec[finalIndx] = -self.abs_elval if minus_sign else self.abs_elval

        return outvec

    def probability(self, state):
        scratch = _np.empty(self.dim, 'd')
        Edense = self.to_dense(scratch)
        return _np.dot(Edense, state.base)  # not vdot b/c data is *real*

#    def alt_to_dense(self):  # Useful
#        if self._evotype == "densitymx":
#            factor_dim = 4
#            v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
#            v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
#        elif self._evotype in ("statevec", "stabilizer", "chp"):
#            factor_dim = 2
#            v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
#            v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
#        elif self._evotype in ("svterm", "cterm"):
#            raise NotImplementedError("to_dense() is not implemented for evotype %s!" %
#                                      self._evotype)
#        else: raise ValueError("Invalid `evotype`: %s" % self._evotype)
#
#        v = (v0, v1)
#
#        if _fastcalc is None:  # do it the slow way using numpy
#            return _functools.reduce(_np.kron, [v[i] for i in self._zvals])
#        else:
#            typ = 'd' if self._evotype == "densitymx" else complex
#            fast_kron_array = _np.ascontiguousarray(
#                _np.empty((len(self._zvals), factor_dim), typ))
#            fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(self._zvals), _np.int64))
#            for i, zi in enumerate(self._zvals):
#                fast_kron_array[i, :] = v[zi]
#            ret = _np.ascontiguousarray(_np.empty(factor_dim**len(self._zvals), typ))
#            if self._evotype == "densitymx":
#                _fastcalc.fast_kron(ret, fast_kron_array, fast_kron_factordims)
#            else:
#                _fastcalc.fast_kron_complex(ret, fast_kron_array, fast_kron_factordims)
#            return ret


class EffectRepTensorProduct(EffectRep):

    def __init__(self, povm_factors, effect_labels, state_space):
        #Arrays for speeding up kron product in effect reps
        max_factor_dim = max(fct.dim for fct in povm_factors)
        kron_array = _np.ascontiguousarray(
            _np.empty((len(povm_factors), max_factor_dim), 'd'))
        factordims = _np.ascontiguousarray(
            _np.array([fct.dim for fct in povm_factors], _np.int64))

        #REMOVE
        #rep = replib.DMEffectRepTensorProd(self._fast_kron_array, self._fast_kron_factordims,
        #                                   len(self.factors), self._fast_kron_array.shape[1], dim)
        #def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        #int dim = _np.product(factor_dims) -- just send as argument for speed?
        #assert(dim == _np.product(kron_factor_dims))

        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factordims
        self.nfactors = len(self.povm_factors)
        self.max_factor_dim = max_factor_dim  # Unused
        state_space = _StateSpace.cast(state_space)
        assert(_np.product(factordims) == state_space.dim)
        super(EffectRepTensorProduct, self).__init__(state_space)
        self.factor_effects_have_changed()

    #TODO: fix this:
    #def __reduce__(self):
    #    return (EffectRepTensorProduct,
    #            (self.kron_array, self.factor_dims, self.nfactors, self.max_factor_dim, self.dim))

    def to_dense(self, outvec):
        N = self.dim
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

    def probability(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, 'd')
        Edense = self.to_dense(scratch)
        return _np.dot(Edense, state.base)  # not vdot b/c data is *real*

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self.factor_dims, self.effect_labels)):
            self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense()

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps

    #def to_dense(self):
    #    if len(self.factors) == 0: return _np.empty(0, complex if self._evotype == "statevec" else 'd')
    #        #NOTE: moved a fast version of to_dense to replib - could use that if need a fast to_dense call...
    #
    #        factorPOVMs = self.factors
    #        ret = factorPOVMs[0][self.effectLbls[0]].to_dense()
    #        for i in range(1, len(factorPOVMs)):
    #            ret = _np.kron(ret, factorPOVMs[i][self.effectLbls[i]].to_dense())
    #        return ret
    #    elif self._evotype == "stabilizer":
    #        # each factor is a StabilizerEffectVec
    #        raise ValueError("Cannot convert Stabilizer tensor product effect to an array!")
    #    # should be using effect.outcomes property...
    #    else:  # self._evotype in ("svterm","cterm")
    #        raise NotImplementedError("to_dense() not implemented for %s evolution type" % self._evotype)


class EffectRepComposed(EffectRep):
    def __init__(self, op_rep, effect_rep, op_id):
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.op_id = op_id
        super(EffectRepComposed, self).__init__(effect_rep.state_space)

    def __reduce__(self):
        return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id))

    def probability(self, state):
        state = self.op_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.probability(state)
