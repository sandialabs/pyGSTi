"""Defines Python-version calculation "representation" objects"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import sys
import time as _time
import math as _math
import numpy as _np
import scipy.sparse as _sps
import itertools as _itertools
import functools as _functools

from ...tools import mpitools as _mpit
from ...tools import slicetools as _slct
from ...tools import matrixtools as _mt
from ...tools import listtools as _lt
from ...tools import optools as _gt
from ...tools.matrixtools import _fas

from scipy.sparse.linalg import LinearOperator

LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


# DEBUG!!!
DEBUG_FCOUNT = 0


class DMStateRep(object):
    def __init__(self, data, reducefix=0):
        assert(data.dtype == _np.dtype('d'))
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMStateRep, (self.base, reducefix))

    def copy_from(self, other):
        self.base = other.base.copy()

    def to_dense(self):
        return self.base

    @property
    def dim(self):
        return len(self.base)

    def __str__(self):
        return str(self.base)


class DMEffectRep(object):
    def __init__(self, dim):
        self.dim = dim

    def probability(self, state):
        raise NotImplementedError()


class DMEffectRepDense(DMEffectRep):
    def __init__(self, data, reducefix=0):
        assert(data.dtype == _np.dtype('d'))
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        super(DMEffectRepDense, self).__init__(len(self.base))

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMEffectRepDense, (self.base, reducefix))

    def probability(self, state):
        # can assume state is a DMStateRep
        return _np.dot(self.base, state.base)  # not vdot b/c *real* data


class DMEffectRepTensorProd(DMEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim  # Unused
        super(DMEffectRepTensorProd, self).__init__(dim)

    def __reduce__(self):
        return (DMEffectRepTensorProd,
                (self.kron_array, self.factor_dims, self.nfactors, self.max_factor_dim, self.dim))

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


class DMEffectRepComputational(DMEffectRep):
    def __init__(self, zvals, dim):
        # int dim = 4**len(zvals) -- just send as argument for speed?
        assert(dim == 4**len(zvals))
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

        super(DMEffectRepComputational, self).__init__(dim)

    def __reduce__(self):
        return (DMEffectRepComputational, (self.zvals, self.dim))

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


class DMEffectRepErrgen(DMEffectRep):  # TODO!! Need to make SV version
    def __init__(self, errgen_oprep, effect_rep, errgen_id):
        dim = effect_rep.dim
        self.errgen_rep = errgen_oprep
        self.effect_rep = effect_rep
        self.errgen_id = errgen_id
        super(DMEffectRepErrgen, self).__init__(dim)

    def __reduce__(self):
        return (DMEffectRepErrgen, (self.errgen_rep, self.effect_rep, self.errgen_id))

    def probability(self, state):
        state = self.errgen_rep.acton(state)  # *not* acton_adjoint
        return self.effect_rep.probability(state)


class DMOpRep(object):
    def __init__(self, dim):
        self.dim = dim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = DMStateRep(_np.ascontiguousarray(v, 'd'))
            return self.acton(in_state).to_dense()

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = DMStateRep(_np.ascontiguousarray(v, 'd'))
            return self.adjoint_acton(in_state).to_dense()
        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?


class DMOpRepDense(DMOpRep):
    def __init__(self, data, reducefix=0):
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        super(DMOpRepDense, self).__init__(self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMOpRepDense, (self.base, reducefix))

    def acton(self, state):
        return DMStateRep(_np.dot(self.base, state.base))

    def adjoint_acton(self, state):
        return DMStateRep(_np.dot(self.base.T, state.base))  # no conjugate b/c *real* data

    def __str__(self):
        return "DMOpRepDense:\n" + str(self.base)


class DMOpRepEmbedded(DMOpRep):
    def __init__(self, embedded_op, num_basis_els, action_inds,
                 blocksizes, embedded_dim, ncomponents_in_active_block,
                 active_block_index, nblocks, dim):

        self.embedded = embedded_op
        self.num_basis_els = num_basis_els
        self.action_inds = action_inds
        self.blocksizes = blocksizes

        num_basis_els_noop_blankaction = num_basis_els.copy()
        for i in action_inds: num_basis_els_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in num_basis_els_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(num_basis_els[1:]))))), _np.int64)
        self.basisInds_action = [list(range(num_basis_els[i])) for i in action_inds]

        self.embeddedDim = embedded_dim
        self.ncomponents = ncomponents_in_active_block
        self.active_block_index = active_block_index
        self.nblocks = nblocks
        self.offset = sum(blocksizes[0:active_block_index])
        super(DMOpRepEmbedded, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepEmbedded, (self.embedded,
                                  self.num_basis_els, self.action_inds,
                                  self.blocksizes, self.embeddedDim,
                                  self.ncomponents, self.active_block_index,
                                  self.nblocks, self.dim))

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.active_block_index:
                output_state.base[offset:offset + blockSize] = state.base[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = DMStateRep(_np.zeros(state.base.shape, 'd'))
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        #print("DB REPLIB ACTON: ",self.basisInds_noop_blankaction)
        #print("DB REPLIB ACTON: ",self.basisInds_action)
        #print("DB REPLIB ACTON: ",self.multipliers)
        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep(state.base[inds])
            embedded_outstate = self.embedded.acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = DMStateRep(_np.zeros(state.base.shape, 'd'))
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep(state.base[inds])
            embedded_outstate = self.embedded.adjoint_acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class DMOpRepComposed(DMOpRep):
    def __init__(self, factor_op_reps, dim):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_op_reps
        super(DMOpRepComposed, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepComposed, (self.factor_reps, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factor_reps:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factor_reps):
            state = gate.adjoint_acton(state)
        return state

    def reinit_factor_op_reps(self, new_factor_op_reps):
        self.factor_reps = new_factor_op_reps


class DMOpRepSum(DMOpRep):
    def __init__(self, factor_reps, dim):
        #assert(len(factor_reps) > 0), "Summed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(DMOpRepSum, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepSum, (self.factor_reps, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = DMStateRep(_np.zeros(state.base.shape, 'd'))
        for f in self.factor_reps:
            output_state.base += f.acton(state).base
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = DMStateRep(_np.zeros(state.base.shape, 'd'))
        for f in self.factor_reps:
            output_state.base += f.adjoint_acton(state).base
        return output_state


class DMOpRepExponentiated(DMOpRep):
    def __init__(self, exponentiated_op_rep, power, dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(DMOpRepExponentiated, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepExponentiated, (self.exponentiated_op, self.power, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.adjoint_acton(state)
        return state


class DMOpRepLindblad(DMOpRep):
    def __init__(self, errgen_rep,
                 mu, eta, m_star, s, unitarypost_data,
                 unitarypost_indices, unitarypost_indptr):
        dim = errgen_rep.dim
        self.errgen_rep = errgen_rep
        if len(unitarypost_data) > 0:  # (nnz > 0)
            self.unitary_postfactor = _sps.csr_matrix(
                (unitarypost_data, unitarypost_indices,
                 unitarypost_indptr), shape=(dim, dim))
        else:
            self.unitary_postfactor = None  # no unitary postfactor

        self.mu = mu
        self.eta = eta
        self.m_star = m_star
        self.s = s
        super(DMOpRepLindblad, self).__init__(dim)

    def set_exp_params(self, mu, eta, m_star, s):
        self.mu = mu
        self.eta = eta
        self.m_star = m_star
        self.s = s

    def get_exp_params(self):
        return (self.mu, self.eta, self.m_star, self.s)

    def __reduce__(self):
        if self.unitary_postfactor is None:
            return (DMOpRepLindblad, (self.errgen_rep, self.mu, self.eta, self.m_star, self.s,
                                      _np.empty(0, 'd'), _np.empty(0, _np.int64), _np.zeros(1, _np.int64)))
        else:
            return (DMOpRepLindblad, (self.errgen_rep, self.mu, self.eta, self.m_star, self.s,
                                      self.unitary_postfactor.data, self.unitary_postfactor.indices,
                                      self.unitary_postfactor.indptr))

    def acton(self, state):
        """ Act this gate map on an input state """
        if self.unitary_postfactor is not None:
            statedata = self.unitary_postfactor.dot(state.base)
        else:
            statedata = state.base

        tol = 1e-16  # 2^-53 (=Scipy default) -- TODO: make into an arg?
        A = self.errgen_rep.aslinearoperator()  # ~= a sparse matrix for call below
        statedata = _mt._custom_expm_multiply_simple_core(
            A, statedata, self.mu, self.m_star, self.s, tol, self.eta)
        return DMStateRep(statedata)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for sparse Lindblad LinearOperator Reps yet.")


class DMOpRepSparse(DMOpRep):
    def __init__(self, a_data, a_indices, a_indptr):
        dim = len(a_indptr) - 1
        self.A = _sps.csr_matrix((a_data, a_indices, a_indptr), shape=(dim, dim))
        super(DMOpRepSparse, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepSparse, (self.A.data, self.A.indices, self.A.indptr))

    @property
    def data(self):
        return self.A.data

    @property
    def indices(self):
        return self.A.indices

    @property
    def indptr(self):
        return self.A.indptr

    def acton(self, state):
        """ Act this gate map on an input state """
        return DMStateRep(self.A.dot(state.base))

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        Aadj = self.A.conjugate(copy=True).transpose()
        return DMStateRep(Aadj.dot(state.base))


# State vector (SV) propagation wrapper classes
class SVStateRep(object):
    def __init__(self, data, reducefix=0):
        assert(data.dtype == _np.dtype(complex))
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVStateRep, (self.base, reducefix))

    def copy_from(self, other):
        self.base = other.base.copy()

    @property
    def dim(self):
        return len(self.base)

    def to_dense(self):
        return self.base

    def __str__(self):
        return str(self.base)


class SVEffectRep(object):
    def __init__(self, dim):
        self.dim = dim

    def probability(self, state):
        return abs(self.amplitude(state))**2

    def amplitude(self, state):
        raise NotImplementedError()


class SVEffectRepDense(SVEffectRep):
    def __init__(self, data, reducefix=0):
        assert(data.dtype == _np.dtype(complex))
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        super(SVEffectRepDense, self).__init__(len(self.base))

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVEffectRepDense, (self.base, reducefix))

    def amplitude(self, state):
        # can assume state is a SVStateRep
        return _np.vdot(self.base, state.base)  # (or just 'dot')


class SVEffectRepTensorProd(SVEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim  # Unused
        super(SVEffectRepTensorProd, self).__init__(dim)

    def __reduce__(self):
        return (SVEffectRepTensorProd, (self.kron_array, self.factor_dims,
                                        self.nfactors, self.max_factor_dim, self.dim))

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

    def amplitude(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.to_dense(scratch)
        return _np.vdot(Edense, state.base)


class SVEffectRepComputational(SVEffectRep):
    def __init__(self, zvals, dim):
        # int dim = 4**len(zvals) -- just send as argument for speed?
        assert(dim == 2**len(zvals))
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
        for k, v in enumerate(zvals):
            assert(v in (0, 1)), "zvals must contain only 0s and 1s"
            self.nonzero_index += base * v
            base //= 2  # or right shift?
        super(SVEffectRepComputational, self).__init__(dim)

    def __reduce__(self):
        return (SVEffectRepComputational, (self.zvals, self.dim))

    def to_dense(self, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if not trust_outvec_sparsity:
            outvec[:] = 0  # reset everything to zero
        outvec[self.nonzero_index] = 1.0
        return outvec

    def amplitude(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.to_dense(scratch)
        return _np.vdot(Edense, state.base)


class SVOpRep(object):
    def __init__(self, dim):
        self.dim = dim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()


class SVOpRepDense(SVOpRep):
    def __init__(self, data, reducefix=0):
        if reducefix == 0:
            self.base = data
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False
        super(SVOpRepDense, self).__init__(self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (SVOpRepDense, (self.base, reducefix))

    def acton(self, state):
        return SVStateRep(_np.dot(self.base, state.base))

    def adjoint_acton(self, state):
        return SVStateRep(_np.dot(_np.conjugate(self.base.T), state.base))

    def __str__(self):
        return "SVOpRepDense:\n" + str(self.base)


class SVOpRepEmbedded(SVOpRep):
    # exactly the same as DM case
    def __init__(self, embedded_op, num_basis_els, action_inds,
                 blocksizes, embedded_dim, ncomponents_in_active_block,
                 active_block_index, nblocks, dim):

        self.embedded = embedded_op
        self.num_basis_els = num_basis_els
        self.action_inds = action_inds
        self.blocksizes = blocksizes

        num_basis_els_noop_blankaction = num_basis_els.copy()
        for i in action_inds: num_basis_els_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in num_basis_els_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(num_basis_els[1:]))))), _np.int64)
        self.basisInds_action = [list(range(num_basis_els[i])) for i in action_inds]

        self.embeddedDim = embedded_dim
        self.ncomponents = ncomponents_in_active_block
        self.active_block_index = active_block_index
        self.nblocks = nblocks
        self.offset = sum(blocksizes[0:active_block_index])
        super(SVOpRepEmbedded, self).__init__(dim)

    def __reduce__(self):
        return (DMOpRepEmbedded, (self.embedded,
                                  self.num_basis_els, self.action_inds,
                                  self.blocksizes, self.embeddedDim,
                                  self.ncomponents, self.active_block_index,
                                  self.nblocks, self.dim))

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.active_block_index:
                output_state.base[offset:offset + blockSize] = state.base[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = SVStateRep(_np.zeros(state.base.shape, complex))
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep(state.base[inds])
            embedded_outstate = self.embedded.acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = SVStateRep(_np.zeros(state.base.shape, complex))
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep(state.base[inds])
            embedded_outstate = self.embedded.adjoint_acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class SVOpRepComposed(SVOpRep):
    # exactly the same as DM case
    def __init__(self, factor_op_reps, dim):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factors_reps = factor_op_reps
        super(SVOpRepComposed, self).__init__(dim)

    def __reduce__(self):
        return (SVOpRepComposed, (self.factor_reps, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factor_reps:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factor_reps):
            state = gate.adjoint_acton(state)
        return state

    def reinit_factor_op_reps(self, new_factor_op_reps):
        self.factors_reps = new_factor_op_reps


class SVOpRepSum(SVOpRep):
    # exactly the same as DM case
    def __init__(self, factor_reps, dim):
        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(SVOpRepSum, self).__init__(dim)

    def __reduce__(self):
        return (SVOpRepSum, (self.factor_reps, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = SVStateRep(_np.zeros(state.base.shape, complex))
        for f in self.factor_reps:
            output_state.base += f.acton(state).base
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = SVStateRep(_np.zeros(state.base.shape, complex))
        for f in self.factor_reps:
            output_state.base += f.adjoint_acton(state).base
        return output_state


class SVOpRepExponentiated(SVOpRep):
    def __init__(self, exponentiated_op_rep, power, dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(SVOpRepExponentiated, self).__init__(dim)

    def __reduce__(self):
        return (SVOpRepExponentiated, (self.exponentiated_op, self.power, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.adjoint_acton(state)
        return state


# Stabilizer state (SB) propagation wrapper classes
class SBStateRep(object):
    def __init__(self, smatrix, pvectors, amps):
        from ..stabilizer import StabilizerFrame as _StabilizerFrame
        self.sframe = _StabilizerFrame(smatrix, pvectors, amps)
        # just rely on StabilizerFrame class to do all the heavy lifting...

    def __reduce__(self):
        return (SBStateRep, (self.sframe.s, self.sframe.ps, self.sframe.a))

    @property
    def smatrix(self):
        return self.sframe.s

    @property
    def pvectors(self):
        return self.sframe.ps

    @property
    def amps(self):
        return self.sframe.a

    @property
    def nqubits(self):
        return self.sframe.n

    @property
    def dim(self):
        return 2**self.nqubits  # assume "unitary evolution"-type mode

    def copy(self):
        cpy = SBStateRep(_np.zeros((0, 0), _np.int64), None, None)  # makes a dummy cpy.sframe
        cpy.sframe = self.sframe.copy()  # a legit copy *with* qubit filers copied too
        return cpy

    def __str__(self):
        return "SBStateRep:\n" + str(self.sframe)


class SBEffectRep(object):
    def __init__(self, zvals):
        self.zvals = zvals

    def __reduce__(self):
        return (SBEffectRep, (self.zvals,))

    @property
    def nqubits(self):
        return len(self.zvals)

    @property
    def dim(self):
        return 2**self.nqubits  # assume "unitary evolution"-type mode

    def probability(self, state):
        return state.sframe.measurement_probability(self.zvals, check=True)  # use check for now?

    def amplitude(self, state):
        return state.sframe.extract_amplitude(self.zvals)


class SBOpRep(object):
    def __init__(self, n):
        self.n = n  # number of qubits

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    @property
    def nqubits(self):
        return self.n

    @property
    def dim(self):
        return 2**(self.n)  # assume "unitary evolution"-type mode


class SBOpRepEmbedded(SBOpRep):
    def __init__(self, embedded_op, n, qubits):
        self.embedded = embedded_op
        self.qubits = qubits  # qubit *indices*
        super(SBOpRepEmbedded, self).__init__(n)

    def __reduce__(self):
        return (SBOpRepEmbedded, (self.embedded, self.n, self.qubits))

    def acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubits)
        outstate = self.embedded.acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate

    def adjoint_acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubits)
        outstate = self.embedded.adjoint_acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate


class SBOpRepComposed(SBOpRep):
    # exactly the same as DM case except .dim -> .n
    def __init__(self, factor_op_reps, n):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_op_reps
        super(SBOpRepComposed, self).__init__(n)

    def __reduce__(self):
        return (SBOpRepComposed, (self.factor_reps, self.n))

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factor_reps:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factor_reps):
            state = gate.adjoint_acton(state)
        return state


class SBOpRepSum(SBOpRep):
    # exactly the same as DM case except .dim -> .n
    def __init__(self, factor_reps, n):
        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(SBOpRepSum, self).__init__(n)

    def __reduce__(self):
        return (SBOpRepSum, (self.factor_reps, self.n))

    def acton(self, state):
        """ Act this gate map on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()


class SBOpRepExponentiated(SBOpRep):
    def __init__(self, exponentiated_op_rep, power, n):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(SBOpRepExponentiated, self).__init__(n)

    def __reduce__(self):
        return (SBOpRepExponentiated, (self.exponentiated_op, self.power, self.n))

    def acton(self, state):
        """ Act this gate map on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.power):
            state = self.exponentiated_op.adjoint_acton(state)
        return state


class SBOpRepClifford(SBOpRep):
    def __init__(self, smatrix, svector, smatrix_inv, svector_inv, unitary):
        self.smatrix = smatrix
        self.svector = svector
        self.smatrix_inv = smatrix_inv
        self.svector_inv = svector_inv
        self.unitary = unitary
        super(SBOpRepClifford, self).__init__(smatrix.shape[0] // 2)

    def __reduce__(self):
        return (SBOpRepClifford, (self.smatrix, self.svector, self.smatrix_inv, self.svector_inv, self.unitary))

    @property
    def unitary_dagger(self):
        return _np.conjugate(self.unitary.T)

    def acton(self, state):
        """ Act this gate map on an input state """
        state = state.copy()  # (copies any qubit filters in .sframe too)
        state.sframe.clifford_update(self.smatrix, self.svector, self.unitary)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        # Note: cliffords are unitary, so adjoint == inverse
        state = state.copy()  # (copies any qubit filters in .sframe too)
        state.sframe.clifford_update(self.smatrix_inv, self.svector_inv,
                                     _np.conjugate(self.unitary.T))
        return state


# Other classes
class PolynomialRep(dict):
    """
    Representation class for a polynomial.

    This is similar to a full Polynomial
    dictionary, but lacks some functionality and is optimized for computation
    speed.  In particular, the keys of this dict are not tuples of variable
    indices (as in Polynomial) but simple integers encoded from such tuples.
    To perform this mapping, one must specify a maximum order and number of
    variables.
    """

    def __init__(self, int_coeff_dict, max_num_vars, vindices_per_int):
        """
        Create a new PolynomialRep object.

        Parameters
        ----------
        int_coeff_dict : dict
            A dictionary of coefficients whose keys are already-encoded
            integers corresponding to variable-index-tuples (i.e poly
            terms).

        max_num_vars : int
            The maximum number of variables allowed.  For example, if
            set to 2, then only "x0" and "x1" are allowed to appear
            in terms.
        """

        self.max_num_vars = max_num_vars
        self.vindices_per_int = vindices_per_int

        super(PolynomialRep, self).__init__()
        if int_coeff_dict is not None:
            self.update(int_coeff_dict)

    def reinit(self, int_coeff_dict):
        """ TODO: docstring """
        self.clear()
        self.update(int_coeff_dict)

    def mapvec_indices_inplace(self, mapfn_as_vector):
        new_items = {}
        for k, v in self.items():
            new_vinds = tuple((mapfn_as_vector[j] for j in self._int_to_vinds(k)))
            new_items[self._vinds_to_int(new_vinds)] = v
        self.clear()
        self.update(new_items)

    def copy(self):
        """
        Make a copy of this polynomial representation.

        Returns
        -------
        PolynomialRep
        """
        return PolynomialRep(self, self.max_num_vars, self.vindices_per_int)  # construct expects "int" keys

    def abs(self):
        """
        Return a polynomial whose coefficents are the absolute values of this PolynomialRep's coefficients.

        Returns
        -------
        PolynomialRep
        """
        result = {k: abs(v) for k, v in self.items()}
        return PolynomialRep(result, self.max_num_vars, self.vindices_per_int)

    @property
    def int_coeffs(self):  # so we can convert back to python Polys
        """ The coefficient dictionary (with encoded integer keys) """
        return dict(self)  # for compatibility w/C case which can't derive from dict...

    #UNUSED TODO REMOVE
        #def map_indices_inplace(self, mapfn):
    #    """
    #    Map the variable indices in this `PolynomialRep`.
    #    This allows one to change the "labels" of the variables.
    #
    #    Parameters
    #    ----------
    #    mapfn : function
    #        A single-argument function that maps old variable-index tuples
    #        to new ones.  E.g. `mapfn` might map `(0,1)` to `(10,11)` if
    #        we were increasing each variable index by 10.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    new_items = {self._vinds_to_int(mapfn(self._int_to_vinds(k))): v
    #                 for k, v in self.items()}
    #    self.clear()
    #    self.update(new_items)
    #
    #def set_maximums(self, max_num_vars=None):
    #    """
    #    Alter the maximum order and number of variables (and hence the
    #    tuple-to-int mapping) for this polynomial representation.
    #
    #    Parameters
    #    ----------
    #    max_num_vars : int
    #        The maximum number of variables allowed.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    coeffs = {self._int_to_vinds(k): v for k, v in self.items()}
    #    if max_num_vars is not None: self.max_num_vars = max_num_vars
    #    int_coeffs = {self._vinds_to_int(k): v for k, v in coeffs.items()}
    #    self.clear()
    #    self.update(int_coeffs)

    def _vinds_to_int(self, vinds):
        """ Maps tuple of variable indices to encoded int """
        ints_in_key = int(_np.ceil(len(vinds) / self.vindices_per_int))

        ret_tup = []
        for k in range(ints_in_key):
            ret = 0; m = 1
            # last tuple index is most significant
            for i in vinds[k * self.vindices_per_int:(k + 1) * self.vindices_per_int]:
                assert(i < self.max_num_vars), "Variable index exceed maximum!"
                ret += (i + 1) * m
                m *= self.max_num_vars + 1
            assert(ret >= 0), "vinds = %s -> %d!!" % (str(vinds), ret)
            ret_tup.append(ret)
        return tuple(ret_tup)

    def _int_to_vinds(self, indx_tup):
        """ Maps encoded "int" to tuple of variable indices """
        ret = []
        #DB: cnt = 0; orig = indx
        for indx in indx_tup:
            while indx != 0:
                nxt = indx // (self.max_num_vars + 1)
                i = indx - nxt * (self.max_num_vars + 1)
                ret.append(i - 1)
                indx = nxt
                #DB: cnt += 1
                #DB: if cnt > 50: print("VINDS iter %d - indx=%d (orig=%d, nv=%d)" % (cnt,indx,orig,self.max_num_vars))
        return tuple(sorted(ret))

    #UNUSED TODO REMOVE
    #def deriv(self, wrt_param):
    #    """
    #    Take the derivative of this polynomial representation with respect to
    #    the single variable `wrt_param`.
    #
    #    Parameters
    #    ----------
    #    wrt_param : int
    #        The variable index to differentiate with respect to (can be
    #        0 to the `max_num_vars-1` supplied to `__init__`.
    #
    #    Returns
    #    -------
    #    PolynomialRep
    #    """
    #    dcoeffs = {}
    #    for i, coeff in self.items():
    #        ivar = self._int_to_vinds(i)
    #        cnt = float(ivar.count(wrt_param))
    #        if cnt > 0:
    #            l = list(ivar)
    #            del l[ivar.index(wrt_param)]
    #            dcoeffs[tuple(l)] = cnt * coeff
    #    int_dcoeffs = {self._vinds_to_int(k): v for k, v in dcoeffs.items()}
    #    return PolynomialRep(int_dcoeffs, self.max_num_vars, self.vindices_per_int)

    #def evaluate(self, variable_values):
    #    """
    #    Evaluate this polynomial at the given variable values.
    #
    #    Parameters
    #    ----------
    #    variable_values : iterable
    #        The values each variable will be evaluated at.  Must have
    #        length at least equal to the number of variables present
    #        in this `PolynomialRep`.
    #
    #    Returns
    #    -------
    #    float or complex
    #    """
    #    #FUTURE and make this function smarter (Russian peasant)?
    #    ret = 0
    #    for i, coeff in self.items():
    #        ivar = self._int_to_vinds(i)
    #        ret += coeff * _np.product([variable_values[i] for i in ivar])
    #    return ret

    def compact_complex(self):
        """
        Returns a compact representation of this polynomial as a
        `(variable_tape, coefficient_tape)` 2-tuple of 1D nupy arrays.
        The coefficient tape is *always* a complex array, even if
        none of the polynomial's coefficients are complex.

        Such compact representations are useful for storage and later
        evaluation, but not suited to polynomial manipulation.

        Returns
        -------
        vtape : numpy.ndarray
            A 1D array of integers (variable indices).
        ctape : numpy.ndarray
            A 1D array of *complex* coefficients.
        """
        nTerms = len(self)
        vinds = {i: self._int_to_vinds(i) for i in self.keys()}
        nVarIndices = sum(map(len, vinds.values()))
        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64)  # "variable" tape
        ctape = _np.empty(nTerms, complex)  # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i += 1
        for iTerm, k in enumerate(sorted(self.keys())):
            v = vinds[k]  # so don't need to compute self._int_to_vinds(k)
            l = len(v)
            ctape[iTerm] = self[k]
            vtape[i] = l; i += 1
            vtape[i:i + l] = v; i += l
        assert(i == len(vtape)), "Logic Error!"
        return vtape, ctape

    def compact_real(self):
        """
        Returns a real representation of this polynomial as a
        `(variable_tape, coefficient_tape)` 2-tuple of 1D nupy arrays.
        The coefficient tape is *always* a complex array, even if
        none of the polynomial's coefficients are complex.

        Such compact representations are useful for storage and later
        evaluation, but not suited to polynomial manipulation.

        Returns
        -------
        vtape : numpy.ndarray
            A 1D array of integers (variable indices).
        ctape : numpy.ndarray
            A 1D array of *real* coefficients.
        """
        nTerms = len(self)
        vinds = {i: self._int_to_vinds(i) for i in self.keys()}
        nVarIndices = sum(map(len, vinds.values()))
        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64)  # "variable" tape
        ctape = _np.empty(nTerms, complex)  # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i += 1
        for iTerm, k in enumerate(sorted(self.keys())):
            v = vinds[k]  # so don't need to compute self._int_to_vinds(k)
            l = len(v)
            ctape[iTerm] = self[k]
            vtape[i] = l; i += 1
            vtape[i:i + l] = v; i += l
        assert(i == len(vtape)), "Logic Error!"
        return vtape, ctape

    def mult(self, x):
        """
        Returns `self * x` where `x` is another polynomial representation.

        Parameters
        ----------
        x : PolynomialRep

        Returns
        -------
        PolynomialRep
        """
        assert(self.max_num_vars == x.max_num_vars)
        newpoly = PolynomialRep(None, self.max_num_vars, self.vindices_per_int)
        for k1, v1 in self.items():
            for k2, v2 in x.items():
                inds = sorted(self._int_to_vinds(k1) + x._int_to_vinds(k2))
                k = newpoly._vinds_to_int(inds)
                if k in newpoly: newpoly[k] += v1 * v2
                else: newpoly[k] = v1 * v2
        assert(newpoly.degree() <= self.degree() + x.degree())
        return newpoly

    def scale(self, x):
        """
        Performs `self = self * x` where `x` is a scalar.

        Parameters
        ----------
        x : float or complex

        Returns
        -------
        None
        """
        # assume a scalar that can multiply values
        for k in self:
            self[k] *= x

    def add_inplace(self, other):
        """
        Adds `other` into this PolynomialRep.

        Parameters
        ----------
        other : PolynomialRep

        Returns
        -------
        PolynomialRep
        """
        for k, v in other.items():
            try:
                self[k] += v
            except KeyError:
                self[k] = v
        return self

    def add_scalar_to_all_coeffs_inplace(self, x):
        """
        Adds `x` to all of the coefficients in this PolynomialRep.

        Parameters
        ----------
        x : float or complex

        Returns
        -------
        PolynomialRep
        """
        for k in self:
            self[k] += x
        return self

    #UNUSED TODO REMOVE
    #def scalar_mult(self, x):
    #    """
    #    Returns `self * x` where `x` is a scalar.
    #
    #    Parameters
    #    ----------
    #    x : float or complex
    #
    #    Returns
    #    -------
    #    PolynomialRep
    #    """
    #    # assume a scalar that can multiply values
    #    newpoly = self.copy()
    #    newpoly.scale(x)
    #    return newpoly

    def __str__(self):
        def fmt(x):
            if abs(_np.imag(x)) > 1e-6:
                if abs(_np.real(x)) > 1e-6: return "(%.3f+%.3fj)" % (x.real, x.imag)
                else: return "(%.3fj)" % x.imag
            else: return "%.3f" % x.real

        termstrs = []
        sorted_keys = sorted(list(self.keys()))
        for k in sorted_keys:
            vinds = self._int_to_vinds(k)
            varstr = ""; last_i = None; n = 0
            for i in sorted(vinds):
                if i == last_i: n += 1
                elif last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                last_i = i
            if last_i is not None:
                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
            #print("DB: vinds = ",vinds, " varstr = ",varstr)
            if abs(self[k]) > 1e-4:
                termstrs.append("%s%s" % (fmt(self[k]), varstr))
        if len(termstrs) > 0:
            return " + ".join(termstrs)
        else: return "0"

    def __repr__(self):
        return "PolynomialRep[ " + str(self) + " ]"

    def degree(self):
        """ Used for debugging in slowreplib routines only"""
        return max([len(self._int_to_vinds(k)) for k in self.keys()])

    #UNUSED TODO REMOVE
    #def __add__(self, x):
    #    newpoly = self.copy()
    #    if isinstance(x, PolynomialRep):
    #        assert(self.max_num_vars == x.max_num_vars)
    #        for k, v in x.items():
    #            if k in newpoly: newpoly[k] += v
    #            else: newpoly[k] = v
    #    else:  # assume a scalar that can be added to values
    #        for k in newpoly:
    #            newpoly[k] += x
    #    return newpoly
    #
    #def __mul__(self, x):
    #    if isinstance(x, PolynomialRep):
    #        return self.mult_poly(x)
    #    else:  # assume a scalar that can multiply values
    #        return self.mult_scalar(x)
    #
    #def __rmul__(self, x):
    #    return self.__mul__(x)
    #
    #def __pow__(self, n):
    #    ret = PolynomialRep({0: 1.0}, self.max_num_vars, self.vindices_per_int)
    #    cur = self
    #    for i in range(int(_np.floor(_np.log2(n))) + 1):
    #        rem = n % 2  # gets least significant bit (i-th) of n
    #        if rem == 1: ret *= cur  # add current power of x (2^i) if needed
    #        cur = cur * cur  # current power *= 2
    #        n //= 2  # shift bits of n right
    #    return ret
    #
    #def __copy__(self):
    #    return self.copy()
    #
    #def debug_report(self):
    #    actual_max_order = max([len(self._int_to_vinds(k)) for k in self.keys()])
    #    return "PolynomialRep w/max_vars=%d: nterms=%d, actual max-order=%d" % \
    #        (self.max_num_vars, len(self), actual_max_order)
    #


class SVTermRep(object):
    # just a container for other reps (polys, states, effects, and gates)

    @classmethod
    def composed(cls, terms_to_compose, magnitude):
        logmag = _math.log10(magnitude) if magnitude > 0 else -LARGE
        first = terms_to_compose[0]
        coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return SVTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __init__(self, coeff, mag, logmag, pre_state, post_state,
                 pre_effect, post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.magnitude = mag
        self.logmagnitude = logmag
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def set_magnitude(self, mag):
        self.magnitude = mag
        self.logmagnitude = _math.log10(mag) if mag > 0 else -LARGE

    def set_magnitude_only(self, mag):
        self.magnitude = mag

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return SVTermRep(coeff, self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return SVTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)


class SBTermRep(object):
    # exactly the same as SVTermRep
    # just a container for other reps (polys, states, effects, and gates)

    @classmethod
    def composed(cls, terms_to_compose, magnitude):
        logmag = _math.log10(magnitude) if magnitude > 0 else -LARGE
        first = terms_to_compose[0]
        coeffrep = first.coeff
        pre_ops = first.pre_ops[:]
        post_ops = first.post_ops[:]
        for t in terms_to_compose[1:]:
            coeffrep = coeffrep.mult(t.coeff)
            pre_ops += t.pre_ops
            post_ops += t.post_ops
        return SBTermRep(coeffrep, magnitude, logmag, first.pre_state, first.post_state,
                         first.pre_effect, first.post_effect, pre_ops, post_ops)

    def __init__(self, coeff, mag, logmag, pre_state, post_state,
                 pre_effect, post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.magnitude = mag
        self.logmagnitude = logmag
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops

    def set_magnitude(self, mag):
        self.magnitude = mag
        self.logmagnitude = _math.log10(mag) if mag > 0 else -LARGE

    def mapvec_indices_inplace(self, mapvec):
        self.coeff.mapvec_indices_inplace(mapvec)

    def scalar_mult(self, x):
        coeff = self.coeff.copy()
        coeff.scale(x)
        return SBTermRep(coeff, self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)

    def copy(self):
        return SBTermRep(self.coeff.copy(), self.magnitude, self.logmagnitude,
                         self.pre_state, self.post_state, self.pre_effect, self.post_effect,
                         self.pre_ops, self.post_ops)


# No need to create separate classes for floating-pt (vs. polynomial) coeff in Python (no types!)
SVTermDirectRep = SVTermRep
SBTermDirectRep = SBTermRep

## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, operationreps):
    ret = staterep
    for oprep in operationreps:
        ret = oprep.acton(ret)
    return ret


def DM_mapfill_probs_block(fwdsim, mx_to_fill, dest_indices, layout_atom, comm):

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    cacheSize = layout_atom.cache_size

    #Create rhoCache
    rho_cache = [None] * cacheSize  # so we can store (s,p) tuples in cache

    #Get operationreps and ereps now so we don't make unnecessary ._rep references
    rhoreps = {rholbl: fwdsim.model.circuit_layer_operator(rholbl, 'prep')._rep for rholbl in layout_atom.rho_labels}
    operationreps = {gl: fwdsim.model.circuit_layer_operator(gl, 'op')._rep for gl in layout_atom.op_labels}
    effectreps = {i: fwdsim.model.circuit_layer_operator(Elbl, 'povm')._rep for i, Elbl in enumerate(layout_atom.full_effect_labels)}  # cache these in future

    #comm is currently ignored
    #TODO: if layout_atom is split, distribute among processors
    for iDest, iStart, remainder, iCache in layout_atom.table:
        remainder = remainder.circuit_without_povm.layertup

        if iStart is None:  # then first element of remainder is a state prep label
            rholabel = remainder[0]
            init_state = rhoreps[rholabel]
            remainder = remainder[1:]
        else:
            init_state = rho_cache[iStart]  # [:,None]

        #OLD final_state = self.propagate_state(init_state, remainder)
        final_state = propagate_staterep(init_state, [operationreps[gl] for gl in remainder])
        if iCache is not None: rho_cache[iCache] = final_state  # [:,0] #store this state in the cache

        ereps = [effectreps[j] for j in layout_atom.elbl_indices_by_expcircuit[iDest]]
        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]

        for j, erep in zip(final_indices, ereps):
            mx_to_fill[j] = erep.probability(final_state)  # outcome probability


def DM_mapfill_dprobs_block(fwdsim, mx_to_fill, dest_indices, dest_param_indices, layout_atom, param_indices, comm):

    eps = 1e-7  # hardcoded?

    if param_indices is None:
        param_indices = list(range(fwdsim.model.num_params()))
    if dest_param_indices is None:
        dest_param_indices = list(range(_slct.length(param_indices)))

    param_indices = _slct.to_array(param_indices)
    dest_param_indices = _slct.to_array(dest_param_indices)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within mx_to_fill (fpoffset = final parameter offset)
    iParamToFinal = {i: dest_param_indices[st + ii] for ii, i in enumerate(my_param_indices)}

    nEls = layout_atom.num_elements
    probs = _np.empty(nEls, 'd')
    probs2 = _np.empty(nEls, 'd')
    DM_mapfill_probs_block(fwdsim, probs, slice(0, nEls), layout_atom, comm)

    orig_vec = fwdsim.model.to_vector().copy()
    for i in range(fwdsim.model.num_params()):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            fwdsim.model.from_vector(vec, close=True)
            DM_mapfill_probs_block(fwdsim, probs2, slice(0, nEls), layout_atom, subComm)
            _fas(mx_to_fill, [dest_indices, iFinal], (probs2 - probs) / eps)
    fwdsim.model.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of mx_to_fill, so gather together:
    _mpit.gather_slices(all_slices, owners, mx_to_fill, [], axes=1, comm=comm)


def DM_mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                            min_prob_clip_for_weighting, prob_clip_interval, comm):

    def obj_fn(p, f, n_i, n, omitted_p):
        cp = _np.clip(p, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
        v = (p - f) * _np.sqrt(n / cp)

        if omitted_p != 0:
            # if this is the *last* outcome at this time then account for any omitted probability
            omitted_cp = _np.clip(omitted_p, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
            v = _np.sqrt(v**2 + n * omitted_p**2 / omitted_cp)
        return v  # sqrt(the objective function term)  (the qty stored in cache)

    return DM_mapfill_TDterms(fwdsim, obj_fn, array_to_fill, dest_indices, num_outcomes, layout_atom,
                              dataset_rows, comm)


def DM_mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                              min_prob_clip, radius, prob_clip_interval, comm):

    min_p = min_prob_clip; a = radius

    def obj_fn(p, f, n_i, n, omitted_p):
        pos_p = max(p, min_p)
        if n_i != 0:
            freq_term = n_i * (_np.log(f) - 1.0)
        else:
            freq_term = 0.0
        S = -n_i / min_p + n
        S2 = 0.5 * n_i / (min_p**2)
        v = freq_term + -n_i * _np.log(pos_p) + n * pos_p  # dims K x M (K = nSpamLabels, M = n_circuits)

        # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
        v = max(v, 0)

        # quadratic extrapolation of logl at min_p for probabilities < min_p
        if p < min_p:
            v = v + S * (p - min_p) + S2 * (p - min_p)**2

        if n_i == 0:
            if p >= a:
                v = n * p
            else:
                v = n * ((-1.0 / (3 * a**2)) * p**3 + p**2 / a + a / 3.0)
        # special handling for f == 0 terms
        # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

        if omitted_p != 0.0:
            # if this is the *last* outcome at this time then account for any omitted probability
            v += n * omitted_p if omitted_p >= a else \
                n * ((-1.0 / (3 * a**2)) * omitted_p**3 + omitted_p**2 / a + a / 3.0)

        return v  # objective function term (the qty stored in cache)

    return DM_mapfill_TDterms(fwdsim, obj_fn, array_to_fill, dest_indices, num_outcomes, layout_atom,
                              dataset_rows, comm)


def DM_mapfill_TDterms(fwdsim, objfn, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows, comm):

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    cacheSize = layout_atom.cache_size

    EVecs = [fwdsim.model.circuit_layer_operator(elbl, 'povm') for elbl in layout_atom.full_effect_labels]
    #OLD REMOVE: elabels_as_outcomes = [(_gt.effect_label_to_outcome(e),) for e in layout_atom.full_effect_labels]
    #OLD REMOVE: outcome_to_elabel_index = {outcome: i for i, outcome in enumerate(elabels_as_outcomes)}

    assert(cacheSize == 0)  # so all elements have None as start and remainder[0] is a prep label
    #if clip_to is not None:
    #    _np.clip(array_to_fill, clip_to[0], clip_to[1], out=array_to_fill)  # in-place clip

    array_to_fill[dest_indices] = 0.0  # reset destination (we sum into it)

    #comm is currently ignored
    #TODO: if layout_atom is split, distribute among processors
    for iDest, iStart, remainder, iCache in layout_atom.table:
        remainder = remainder.circuit_without_povm.layertup
        assert(iStart is None), "Cannot use trees with max-cache-size > 0 when performing time-dependent calcs!"
        rholabel = remainder[0]; remainder = remainder[1:]
        rhoVec = fwdsim.model.circuit_layer_operator(rholabel, 'prep')
        datarow = dataset_rows[iDest]
        nTotOutcomes = num_outcomes[iDest]

        totalCnts = {}  # TODO defaultdict?
        lastInds = {}; outcome_cnts = {}

        # consolidate multiple outcomes that occur at same time? or sort?
        #CHECK - should this loop filter only outcomes relevant to this expanded circuit (like below)?
        for k, (t0, Nreps) in enumerate(zip(datarow.time, datarow.reps)):
            if t0 in totalCnts:
                totalCnts[t0] += Nreps; outcome_cnts[t0] += 1
            else:
                totalCnts[t0] = Nreps; outcome_cnts[t0] = 1
            lastInds[t0] = k

        elbl_indices = layout_atom.elbl_indices_by_expcircuit[iDest]
        outcomes = layout_atom.outcomes_by_expcircuit[iDest]
        outcome_to_elbl_index = {outcome: elbl_index for outcome, elbl_index in zip(outcomes, elbl_indices)}
        #FUTURE: construct outcome_to_elbl_index dict in layout_atom, so we don't construct it here?
        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]
        elbl_to_final_index = {elbl_index: final_index for elbl_index, final_index in zip(elbl_indices, final_indices)}

        cur_probtotal = 0; last_t = 0
        # consolidate multiple outcomes that occur at same time? or sort?
        for k, (t0, Nreps, outcome) in enumerate(zip(datarow.time, datarow.reps, datarow.outcomes)):
            if outcome not in outcome_to_elbl_index:
                continue  # skip datarow outcomes not for this expanded circuit

            t = t0
            rhoVec.set_time(t)
            rho = rhoVec._rep
            t += rholabel.time

            for gl in remainder:
                op = fwdsim.model.circuit_layer_operator(gl, 'op')
                op.set_time(t); t += gl.time  # time in gate label == gate duration?
                rho = op._rep.acton(rho)

            j = outcome_to_elbl_index[outcome]
            E = EVecs[j]; E.set_time(t)
            p = E._rep.probability(rho)  # outcome probability
            N = totalCnts[t0]
            f = Nreps / N

            if t0 == last_t:
                cur_probtotal += p
            else:
                last_t = t0
                cur_probtotal = p

            omitted_p = 1.0 - cur_probtotal if (lastInds[t0] == k and outcome_cnts[t0] < nTotOutcomes) else 0.0
            # and cur_probtotal < 1.0?

            array_to_fill[elbl_to_final_index[j]] += objfn(p, f, Nreps, N, omitted_p)


def DM_mapfill_TDdchi2_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes, layout_atom,
                             dataset_rows, min_prob_clip_for_weighting, prob_clip_interval, wrt_slice, comm):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm):
        DM_mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, n_outcomes,
                                layout_atom, dataset_rows, min_prob_clip_for_weighting, prob_clip_interval, fill_comm)

    return DM_mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices,
                                     num_outcomes, layout_atom, dataset_rows, fillfn, wrt_slice, comm)


def DM_mapfill_TDdloglpp_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                               layout_atom, dataset_rows, min_prob_clip, radius, prob_clip_interval, wrt_slice, comm):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm):
        DM_mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, n_outcomes,
                                  layout_atom, dataset_rows, min_prob_clip, radius, prob_clip_interval, fill_comm)

    return DM_mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices,
                                     num_outcomes, layout_atom, dataset_rows, fillfn, wrt_slice, comm)


def DM_mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes, layout_atom,
                              dataset_rows, fillfn, wrt_slice, comm):

    eps = 1e-7  # hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(fwdsim.model.num_params()) if (wrt_slice is None) else _slct.indices(wrt_slice)

    nEls = layout_atom.num_elements
    vals = _np.empty(nEls, 'd')
    vals2 = _np.empty(nEls, 'd')
    assert(layout_atom.cache_size == 0)  # so all elements have None as start and remainder[0] is a prep label

    fillfn(vals, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, comm)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

    orig_vec = fwdsim.model.to_vector().copy()
    for i in range(fwdsim.model.num_params()):
        #print("dprobs cache %d of %d" % (i,fwdsim.model.num_params()))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            fwdsim.model.from_vector(vec, close=True)
            fillfn(vals2, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, subComm)
            _fas(array_to_fill, [dest_indices, iFinal], (vals2 - vals) / eps)
    fwdsim.model.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, array_to_fill, [], axes=1, comm=comm)

    #REMOVE
    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (fwdsim.model.num_params(), fwdsim.model.dim, cache_size, len(layout_atom), layout_atom.num_applies(), _time.time()-tStart)) #DEBUG


def SV_prs_as_polys(fwdsim, rholabel, elabels, circuit, comm=None, mem_limit=None, fastmode=True):
    return _prs_as_polys(fwdsim, rholabel, elabels, circuit, comm, mem_limit, fastmode)


def SB_prs_as_polynomials(fwdsim, rholabel, elabels, circuit, comm=None, mem_limit=None, fastmode=True):
    return _prs_as_polys(fwdsim, rholabel, elabels, circuit, comm, mem_limit, fastmode)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_polys(fwdsim, rholabel, elabels, circuit, comm=None, mem_limit=None, fastmode=True):
    """
    Computes polynomials of the probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    Returns
    -------
    list
        A list of PolynomialRep objects, one per element of `elabels`.
    """
    #print("PRS_AS_POLY circuit = ",circuit)
    #print("DB: prs_as_polys(",spamTuple,circuit,fwdsim.max_order,")")

    #NOTE for FUTURE: to adapt this to work with numerical rather than polynomial coeffs:
    # use get_direct_order_terms(order, order_base) w/order_base=0.1(?) instead of taylor_order_terms??
    # below: replace prps with: prs = _np.zeros(len(elabels),complex)  # an array in "bulk" mode
    #  use *= or * instead of .mult( and .scale(
    #  e.g. res = _np.product([f.coeff for f in factors])
    #       res *= (pLeft * pRight)
    # - add assert(_np.linalg.norm(_np.imag(prs)) < 1e-6) at end and return _np.real(prs)

    mpv = fwdsim.model.num_params()  # max_polynomial_vars

    # Construct dict of gate term reps
    distinct_gateLabels = sorted(set(circuit))
    op_term_reps = {glbl:
                    [
                        [t.torep() for t in fwdsim.model.circuit_layer_operator(glbl, 'op').taylor_order_terms(order, mpv)]
                        for order in range(fwdsim.max_order + 1)
                    ] for glbl in distinct_gateLabels}

    #Similar with rho_terms and E_terms, but lists
    rho_term_reps = [[t.torep() for t in fwdsim.model.circuit_layer_operator(rholabel, 'prep').taylor_order_terms(order, mpv)]
                     for order in range(fwdsim.max_order + 1)]

    E_term_reps = []
    E_indices = []
    for order in range(fwdsim.max_order + 1):
        cur_term_reps = []  # the term reps for *all* the effect vectors
        cur_indices = []  # the Evec-index corresponding to each term rep
        for i, elbl in enumerate(elabels):
            term_reps = [t.torep() for t in fwdsim.model.circuit_layer_operator(elbl, 'povm').taylor_order_terms(order, mpv)]
            cur_term_reps.extend(term_reps)
            cur_indices.extend([i] * len(term_reps))
        E_term_reps.append(cur_term_reps)
        E_indices.append(cur_indices)

    ##DEBUG!!!
    #print("DB NEW operation terms = ")
    #for glbl,order_terms in op_term_reps.items():
    #    print("GATE ",glbl)
    #    for i,termlist in enumerate(order_terms):
    #        print("ORDER %d" % i)
    #        for term in termlist:
    #            print("Coeff: ",str(term.coeff))

    #HERE DEBUG!!!
    global DEBUG_FCOUNT
    # db_part_cnt = 0
    # db_factor_cnt = 0
    #print("DB: pr_as_poly for ",str(tuple(map(str,circuit))), " max_order=",fwdsim.max_order)

    prps = [None] * len(elabels)  # an array in "bulk" mode? or Polynomial in "symbolic" mode?
    for order in range(fwdsim.max_order + 1):
        #print("DB: pr_as_poly order=",order)
        # db_npartitions = 0
        for p in _lt.partition_into(order, len(circuit) + 2):  # +2 for SPAM bookends
            #factor_lists = [ fwdsim.sos.operation(glbl).get_order_terms(pi) for glbl,pi in zip(circuit,p) ]
            factor_lists = [rho_term_reps[p[0]]] + \
                           [op_term_reps[glbl][pi] for glbl, pi in zip(circuit, p[1:-1])] + \
                           [E_term_reps[p[-1]]]
            factor_list_lens = list(map(len, factor_lists))
            Einds = E_indices[p[-1]]  # specifies which E-vec index each of E_term_reps[p[-1]] corresponds to

            if any([len(fl) == 0 for fl in factor_lists]): continue

            #print("DB partition = ",p, "listlens = ",[len(fl) for fl in factor_lists])
            if fastmode:  # filter factor_lists to matrix-compose all length-1 lists
                leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
                rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
                coeffSaved = [None] * (len(factor_lists) - 1)
                last_index = len(factor_lists) - 1

                for incd, fi in _lt.incd_product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(fi)]

                    if incd == 0:  # need to re-evaluate rho vector
                        rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                        for f in factors[0].pre_ops:
                            rhoVecL = f.acton(rhoVecL)
                        leftSaved[0] = rhoVecL

                        rhoVecR = factors[0].post_state
                        for f in factors[0].post_ops:
                            rhoVecR = f.acton(rhoVecR)
                        rightSaved[0] = rhoVecR

                        coeff = factors[0].coeff
                        coeffSaved[0] = coeff
                        incd += 1
                    else:
                        rhoVecL = leftSaved[incd - 1]
                        rhoVecR = rightSaved[incd - 1]
                        coeff = coeffSaved[incd - 1]

                    # propagate left and right states, saving as we go
                    for i in range(incd, last_index):
                        for f in factors[i].pre_ops:
                            rhoVecL = f.acton(rhoVecL)
                        leftSaved[i] = rhoVecL

                        for f in factors[i].post_ops:
                            rhoVecR = f.acton(rhoVecR)
                        rightSaved[i] = rhoVecR

                        coeff = coeff.mult(factors[i].coeff)
                        coeffSaved[i] = coeff

                    # for the last index, no need to save, and need to construct
                    # and apply effect vector

                    #HERE - add something like:
                    #  if factors[-1].opname == cur_effect_opname: (or opint in C-case)
                    #      <skip application of post_ops & preops - just load from (new) saved slot get pLeft & pRight>

                    for f in factors[-1].pre_ops:
                        rhoVecL = f.acton(rhoVecL)
                    E = factors[-1].post_effect  # effect representation
                    pLeft = E.amplitude(rhoVecL)

                    #Same for post_ops and rhoVecR
                    for f in factors[-1].post_ops:
                        rhoVecR = f.acton(rhoVecR)
                    E = factors[-1].pre_effect
                    pRight = _np.conjugate(E.amplitude(rhoVecR))

                    #print("DB PYTHON: final block: pLeft=",pLeft," pRight=",pRight)
                    res = coeff.mult(factors[-1].coeff)
                    res.scale((pLeft * pRight))
                    #print("DB PYTHON: result = ",res)
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx]  # final "factor" index == E-vector index
                    if prps[Ei] is None: prps[Ei] = res
                    else: prps[Ei] += res  # could add_inplace?
                    #print("DB PYTHON: prps[%d] = " % Ei, prps[Ei])

            else:  # non-fast mode
                last_index = len(factor_lists) - 1
                for fi in _itertools.product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(fi)]
                    res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                    pLeft = _unitary_sim_pre(factors, comm, mem_limit)
                    pRight = _unitary_sim_post(factors, comm, mem_limit)
                    # if not self.unitary_evolution else 1.0
                    res.scale((pLeft * pRight))
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx]  # final "factor" index == E-vector index
                    # print("DB: pr_as_poly    ", fi, " coeffs=", [f.coeff for f in factors],
                    #       " pLeft=", pLeft, " pRight=", pRight, "res=", res)
                    if prps[Ei] is None: prps[Ei] = res
                    else: prps[Ei] += res  # add_inplace?
                    #print("DB pr_as_poly   running prps[",Ei,"] =",prps[Ei])

            # #DEBUG!!!
            # db_nfactors = [len(l) for l in factor_lists]
            # db_totfactors = _np.product(db_nfactors)
            # db_factor_cnt += db_totfactors
            # DEBUG_FCOUNT += db_totfactors
            # db_part_cnt += 1
            # print("DB: pr_as_poly   partition=",p,
            #       "(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

    #print("DONE -> FCOUNT=",DEBUG_FCOUNT)
    return prps  # can be a list of polys


def SV_prs_directly(fwdsim, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True, wt_tol=0.0,
                    reset_term_weights=True, debug=None):
    #return _prs_directly(fwdsim, rholabel, elabels, circuit, comm, mem_limit, fastmode)
    raise NotImplementedError("No direct mode yet")


def SB_prs_directly(fwdsim, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True, wt_tol=0.0,
                    reset_term_weights=True, debug=None):
    #return _prs_directly(fwdsim, rholabel, elabels, circuit, comm, mem_limit, fastmode)
    raise NotImplementedError("No direct mode yet")


def SV_refresh_magnitudes_in_repcache(repcache, paramvec):
    from ..opcalc import bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex
    for repcel in repcache.values():
        #repcel = <RepCacheEl?>repcel
        for termrep in repcel[0]:  # first element of tuple contains list of term-reps
            v, c = termrep.coeff.compact_complex()
            coeff_array = _bulk_eval_compact_polynomials_complex(v, c, paramvec, (1,))
            termrep.set_magnitude_only(abs(coeff_array[0]))


def SV_find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, repcache, circuitsetup_cache,
                                         comm=None, mem_limit=None, pathmagnitude_gap=0.0, min_term_mag=0.01,
                                         max_paths=500, threshold_guess=0.0):
    return _find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, repcache, circuitsetup_cache,
                                              comm, mem_limit, pathmagnitude_gap, min_term_mag, max_paths,
                                              threshold_guess)


def SB_find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, repcache, circuitsetup_cache,
                                         comm=None, mem_limit=None, pathmagnitude_gap=0.0, min_term_mag=0.01,
                                         max_paths=500, threshold_guess=0.0):
    return _find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, repcache, circuitsetup_cache,
                                              comm, mem_limit, pathmagnitude_gap, min_term_mag, max_paths,
                                              threshold_guess)


def SV_compute_pruned_path_polynomials_given_threshold(threshold, fwdsim, rholabel, elabels, circuit, repcache,
                                                       circuitsetup_cache, comm=None, mem_limit=None, fastmode=True):
    return _compute_pruned_path_polys_given_threshold(threshold, fwdsim, rholabel, elabels, circuit, repcache,
                                                      circuitsetup_cache, comm, mem_limit, fastmode)


def SB_compute_pruned_path_polynomials_given_threshold(threshold, fwdsim, rholabel, elabels, circuit, repcache,
                                                       circuitsetup_cache, comm=None, mem_limit=None, fastmode=True):
    return _compute_pruned_path_polys_given_threshold(threshold, fwdsim, rholabel, elabels, circuit, repcache,
                                                      circuitsetup_cache, comm, mem_limit, fastmode)


def SV_circuit_achieved_and_max_sopm(fwdsim, rholabel, elabels, circuit, repcache, threshold, min_term_mag):
    """ TODO: docstring """
    mpv = fwdsim.model.num_params()  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model.circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)

    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]

    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model.circuit_layer_operator(rholabel, 'prep')] + [fwdsim.model.circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.product([op.total_term_magnitude() for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model.circuit_layer_operator(elbl, 'povm').total_term_magnitude() for elbl in elabels], 'd')

    mag = _np.zeros(len(elabels), 'd')
    nPaths = _np.zeros(len(elabels), int)

    def count_path(b, mg, incd):
        mag[E_indices[b[-1]]] += mg
        nPaths[E_indices[b[-1]]] += 1

    traverse_paths_upto_threshold(factor_lists, threshold, len(elabels),
                                  foat_indices_per_op, count_path)  # sets mag and nPaths
    return mag, max_sum_of_pathmags

    #threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
    #    factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
    #    initial_threshold=current_threshold, min_threshold=pathmagnitude_gap / 1000.0, max_npaths=max_paths)


global_cnt = 0

#Base case which works for both SV and SB evolution types thanks to Python's duck typing


def _find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, repcache, circuitsetup_cache, comm,
                                       mem_limit, pathmagnitude_gap, min_term_mag, max_paths, threshold_guess):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    circuitsetup_cache : dict, optional
        Dictionary used to cache preparation specific to this function, to
        speed up repeated calls using the same circuit and set of parameters,
        including the same repcache.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    pathmagnitude_gap : float, optional
        The amount less than the perfect sum-of-path-magnitudes that
        is desired.  This sets the target sum-of-path-magnitudes for each
        circuit -- the threshold that determines how many paths are added.

    min_term_mag : float, optional
        A technical parameter to the path pruning algorithm; this value
        sets a threshold for how small a term magnitude (one factor in
        a path magnitude) must be before it is removed from consideration
        entirely (to limit the number of even *potential* paths).  Terms
        with a magnitude lower than this values are neglected.

    max_paths : int, optional
        The maximum number of paths allowed per circuit outcome.

    threshold_guess : float, optional
        In the search for a good pathmagnitude threshold, this value is
        used as the starting point.  If 0.0 is given, a default value is used.

    Returns
    -------
    npaths : int
        the number of paths that were included.
    threshold : float
        the path-magnitude threshold used.
    target_sopm : float
        The desired sum-of-path-magnitudes.  This is `pathmagnitude_gap`
        less than the perfect "all-paths" sum.  This sums together the
        contributions of different effects.
    achieved_sopm : float
        The achieved sum-of-path-magnitudes.  Ideally this would equal
        `target_sopm`. (This also sums together the contributions of
        different effects.)
    """
    if circuitsetup_cache is None: circuitsetup_cache = {}

    if circuit not in circuitsetup_cache:
        circuitsetup_cache[circuit] = create_circuitsetup_cacheel(
            fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, fwdsim.model.num_params())
    rho_term_reps, op_term_reps, E_term_reps, \
        rho_foat_indices, op_foat_indices, E_foat_indices, E_indices = circuitsetup_cache[circuit]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model.circuit_layer_operator(rholabel, 'prep')] + [fwdsim.model.circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.product([op.total_term_magnitude() for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model.circuit_layer_operator(elbl, 'povm').total_term_magnitude() for elbl in elabels], 'd')
    target_sum_of_pathmags = max_sum_of_pathmags - pathmagnitude_gap  # absolute gap
    #target_sum_of_pathmags = max_sum_of_pathmags * (1.0 - pathmagnitude_gap)  # relative gap
    threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
        factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
        initial_threshold=threshold_guess, min_threshold=pathmagnitude_gap / (3.0 * max_paths),  # 3.0 is just heuristic
        max_npaths=max_paths)
    # above takes an array of target pathmags and gives a single threshold that works for all of them (all E-indices)

    # TODO REMOVE
    #print("Threshold = ", threshold, " Paths=", npaths)
    #REMOVE (and global_cnt definition above)
    #global global_cnt
    # print("Threshold = ", threshold, " Paths=", npaths, " tgt=", target_sum_of_pathmags,
    #       "cnt = ", global_cnt)  # , " time=%.3fs" % (_time.time()-t0))
    #global_cnt += 1

    # #DEBUG TODO REMOVE
    # print("---------------------------")
    # print("Path threshold = ",threshold, " max=",max_sum_of_pathmags,
    #       " target=",target_sum_of_pathmags, " achieved=",achieved_sum_of_pathmags)
    # print("nPaths = ",npaths)
    # print("Num high-magnitude (|coeff|>%g, taylor<=%d) terms: %s" \
    #       % (min_term_mag, fwdsim.max_order, str([len(factors) for factors in factor_lists])))
    # print("Num FOAT: ",[len(inds) for inds in foat_indices_per_op])
    # print("---------------------------")

    target_miss = sum(achieved_sum_of_pathmags) - sum(target_sum_of_pathmags + pathmagnitude_gap)
    if target_miss > 1e-5:
        print("Warning: Achieved sum(path mags) exceeds max by ", target_miss, "!!!")

    return sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)


def _compute_pruned_path_polys_given_threshold(threshold, fwdsim, rholabel, elabels, circuit, repcache,
                                               circuitsetup_cache, comm, mem_limit, fastmode):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    circuitsetup_cache : dict, optional
        Dictionary used to cache preparation specific to this function, to
        speed up repeated calls using the same circuit and set of parameters,
        including the same repcache.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    Returns
    -------
    prps : list of PolynomialRep objects
        the polynomials for the requested circuit probabilities, computed by
        selectively summing up high-magnitude paths.
    """
    if circuitsetup_cache is None: circuitsetup_cache = {}

    if circuit not in circuitsetup_cache:
        circuitsetup_cache[circuit] = create_circuitsetup_cacheel(
            fwdsim, rholabel, elabels, circuit, repcache, fwdsim.min_term_mag, fwdsim.model.num_params())
    rho_term_reps, op_term_reps, E_term_reps, \
        rho_foat_indices, op_foat_indices, E_foat_indices, E_indices = circuitsetup_cache[circuit]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    prps = [None] * len(elabels)
    last_index = len(factor_lists) - 1

    #print("T1 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    #fastmode = False  # REMOVE - was used for DEBUG b/c "_ex" path traversal won't always work w/fast mode
    if fastmode == 1:  # fastmode
        leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
        rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
        coeffSaved = [None] * (len(factor_lists) - 1)

        def add_path(b, mag, incd):
            """ Relies on the fact that paths are iterated over in lexographic order, and `incd`
                tells us which index was just incremented (all indices less than this one are
                the *same* as the last call). """
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no
            # ability to cache?)
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]

            if incd == 0:  # need to re-evaluate rho vector
                rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                for f in factors[0].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factors[0].post_state
                for f in factors[0].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[0] = rhoVecR

                coeff = factors[0].coeff
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd - 1]
                rhoVecR = rightSaved[incd - 1]
                coeff = coeffSaved[incd - 1]

            # propagate left and right states, saving as we go
            for i in range(incd, last_index):
                for f in factors[i].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[i] = rhoVecL

                for f in factors[i].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = coeff.mult(factors[i].coeff)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            for f in factors[-1].pre_ops:
                rhoVecL = f.acton(rhoVecL)
            E = factors[-1].post_effect  # effect representation
            pLeft = E.amplitude(rhoVecL)

            #Same for post_ops and rhoVecR
            for f in factors[-1].post_ops:
                rhoVecR = f.acton(rhoVecR)
            E = factors[-1].pre_effect
            pRight = _np.conjugate(E.amplitude(rhoVecR))

            res = coeff.mult(factors[-1].coeff)
            res.scale((pLeft * pRight))
            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index

            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    elif fastmode == 2:  # achieved-SOPM mode
        def add_path(b, mag, incd):
            """Adds in |pathmag| = |prod(factor_coeffs)| for computing achieved SOPM"""
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff.abs() for f in factors])

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    else:
        def add_path(b, mag, incd):
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
            pLeft = _unitary_sim_pre(factors, comm, mem_limit)
            pRight = _unitary_sim_post(factors, comm, mem_limit)
            res.scale((pLeft * pRight))

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

            #print("DB running prps[",Ei,"] =",prps[Ei])

    traverse_paths_upto_threshold(factor_lists, threshold, len(
        elabels), foat_indices_per_op, add_path)  # sets mag and nPaths

    #print("T2 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    #max_degrees = []
    #for i,factors in enumerate(factor_lists):
    #    max_degrees.append(max([f.coeff.degree() for f in factors]))
    #print("Max degrees = ",max_degrees)
    #for Ei,prp in enumerate(prps):
    #    print(Ei,":", prp.debug_report())
    #if db_paramvec is not None:
    #    for Ei,prp in enumerate(prps):
    #        print(Ei," => ", prp.evaluate(db_paramvec))

    #TODO: REMOVE - most of this is solved, but keep it around for another few commits in case we want to refer back to
    #it.  - need to fill in some more details, namely how/where we hold weights and log-weights: in reps? in Term objs?
    #maybe consider Cython version?  need to consider how to perform "fastmode" in this... maybe need to traverse tree
    #in some standard order?  what about having multiple thresholds for the different elabels... it seems good to try to
    #run these calcs in parallel.
    # Note: may only need recusive tree traversal to consider incrementing positions *greater* than or equal to the one
    #  that was just incremented?  (this may enforce some iteration ordering amenable to a fastmode calc)
    # Note2: when all effects have *same* op-part of terms, just different effect vector, then maybe we could split the
    #  effect into an op + effect to better use fastmode calc?  Or maybe if ordering is right this isn't necessary?
    #Add repcache as in cython version -- saves having to *construct* rep objects all the time... just update
    #coefficients when needed instead?

    #... and we're done!

    #TODO: check that prps are PolynomialReps and not Polynomials -- we may have made this change
    # in fastreplib.pyx but forgot it here.
    return prps


def create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv):
    # Construct dict of gate term reps
    mpv = fwdsim.model.num_params()  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model.circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)
    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    return (rho_term_reps, op_term_reps, E_term_reps,
            rho_foat_indices, op_foat_indices, E_foat_indices,
            E_indices)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_pruned_polys(fwdsim, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True,
                         pathmagnitude_gap=0.0, min_term_mag=0.01, max_paths=500, current_threshold=None,
                         compute_polyreps=True):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

        pathmagnitude_gap : float, optional
            The amount less than the perfect sum-of-path-magnitudes that
            is desired.  This sets the target sum-of-path-magnitudes for each
            circuit -- the threshold that determines how many paths are added.

        min_term_mag : float, optional
            A technical parameter to the path pruning algorithm; this value
            sets a threshold for how small a term magnitude (one factor in
            a path magnitude) must be before it is removed from consideration
            entirely (to limit the number of even *potential* paths).  Terms
            with a magnitude lower than this values are neglected.

        current_threshold : float, optional
            If the threshold needed to achieve the desired `pathmagnitude_gap`
            is greater than this value (i.e. if using current_threshold would
            result in *more* paths being computed) then this function will not
            compute any paths and exit early, returning `None` in place of the
            usual list of polynomial representations.

    compute_polyreps: TODO, docstring - whether to just compute sopm or actually compute corresponding polyreps

    Returns
    -------
    prps : list of PolynomialRep objects
        the polynomials for the requested circuit probabilities, computed by
        selectively summing up high-magnitude paths.
    npaths : int
        the number of paths that were included.
    threshold : float
        the path-magnitude threshold used.
    target_sopm : float
        The desired sum-of-path-magnitudes.  This is `pathmagnitude_gap`
        less than the perfect "all-paths" sum.  This sums together the
        contributions of different effects.
    achieved_sopm : float
        The achieved sum-of-path-magnitudes.  Ideally this would equal
        `target_sopm`. (This also sums together the contributions of
        different effects.)
    """
    #t0 = _time.time()
    # Construct dict of gate term reps
    mpv = fwdsim.model.num_params()  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model.circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model.circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)

    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    prps = [None] * len(elabels)

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    last_index = len(factor_lists) - 1

    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model.circuit_layer_operator(rholabel, 'prep')] + [fwdsim.model.circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.product([op.total_term_magnitude() for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model.circuit_layer_operator(elbl, 'povm').total_term_magnitude() for elbl in elabels], 'd')
    target_sum_of_pathmags = max_sum_of_pathmags - pathmagnitude_gap  # absolute gap
    #target_sum_of_pathmags = max_sum_of_pathmags * (1.0 - pathmagnitude_gap)  # relative gap
    threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
        factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
        initial_threshold=current_threshold,
        min_threshold=pathmagnitude_gap / (3.0 * max_paths),  # 3.0 is just heuristic
        max_npaths=max_paths)
    # above takes an array of target pathmags and gives a single threshold that works for all of them (all E-indices)

    #print("Threshold = ", threshold, " Paths=", npaths)
    #REMOVE (and global_cnt definition above)
    #global global_cnt
    # print("Threshold = ", threshold, " Paths=", npaths, " tgt=", target_sum_of_pathmags,
    #       "cnt = ", global_cnt)  # , " time=%.3fs" % (_time.time()-t0))
    #global_cnt += 1

    # no polyreps needed, e.g. just keep existing (cached) polys
    if not compute_polyreps or (current_threshold >= 0 and threshold >= current_threshold):
        return [], sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)

    #print("T1 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    #fastmode = False  # REMOVE - was used for DEBUG b/c "_ex" path traversal won't always work w/fast mode
    if fastmode:
        leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
        rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
        coeffSaved = [None] * (len(factor_lists) - 1)

        def add_path(b, mag, incd):
            """ Relies on the fact that paths are iterated over in lexographic order, and `incd`
                tells us which index was just incremented (all indices less than this one are
                the *same* as the last call). """
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no
            # ability to cache?)
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]

            if incd == 0:  # need to re-evaluate rho vector
                rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                for f in factors[0].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factors[0].post_state
                for f in factors[0].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[0] = rhoVecR

                coeff = factors[0].coeff
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd - 1]
                rhoVecR = rightSaved[incd - 1]
                coeff = coeffSaved[incd - 1]

            # propagate left and right states, saving as we go
            for i in range(incd, last_index):
                for f in factors[i].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[i] = rhoVecL

                for f in factors[i].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = coeff.mult(factors[i].coeff)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            for f in factors[-1].pre_ops:
                rhoVecL = f.acton(rhoVecL)
            E = factors[-1].post_effect  # effect representation
            pLeft = E.amplitude(rhoVecL)

            #Same for post_ops and rhoVecR
            for f in factors[-1].post_ops:
                rhoVecR = f.acton(rhoVecR)
            E = factors[-1].pre_effect
            pRight = _np.conjugate(E.amplitude(rhoVecR))

            res = coeff.mult(factors[-1].coeff)
            res.scale((pLeft * pRight))
            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index

            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    else:
        def add_path(b, mag, incd):
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
            pLeft = _unitary_sim_pre(factors, comm, mem_limit)
            pRight = _unitary_sim_post(factors, comm, mem_limit)
            res.scale((pLeft * pRight))

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res
            #print("DB running prps[",Ei,"] =",prps[Ei])

    traverse_paths_upto_threshold(factor_lists, threshold, len(
        elabels), foat_indices_per_op, add_path)  # sets mag and nPaths

    #print("T2 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    # #DEBUG
    # print("---------------------------")
    # print("Path threshold = ",threshold, " max=",max_sum_of_pathmags,
    #       " target=",target_sum_of_pathmags, " achieved=",achieved_sum_of_pathmags)
    # print("nPaths = ",npaths)
    # print("Num high-magnitude (|coeff|>%g, taylor<=%d) terms: %s" \
    #       % (min_term_mag, fwdsim.max_order, str([len(factors) for factors in factor_lists])))
    # print("Num FOAT: ",[len(inds) for inds in foat_indices_per_op])
    # print("---------------------------")

    #max_degrees = []
    #for i,factors in enumerate(factor_lists):
    #    max_degrees.append(max([f.coeff.degree() for f in factors]))
    #print("Max degrees = ",max_degrees)
    #for Ei,prp in enumerate(prps):
    #    print(Ei,":", prp.debug_report())
    #if db_paramvec is not None:
    #    for Ei,prp in enumerate(prps):
    #        print(Ei," => ", prp.evaluate(db_paramvec))

    #TODO: REMOVE - most of this is solved, but keep it around for another few commits in case we want to refer back to
    #it.  - need to fill in some more details, namely how/where we hold weights and log-weights: in reps? in Term objs?
    #maybe consider Cython version?  need to consider how to perform "fastmode" in this... maybe need to traverse tree
    #in some standard order?  what about having multiple thresholds for the different elabels... it seems good to try to
    #run these calcs in parallel.
    # Note: may only need recusive tree traversal to consider incrementing positions *greater* than or equal to the one
    #  that was just incremented?  (this may enforce some iteration ordering amenable to a fastmode calc)
    # Note2: when all effects have *same* op-part of terms, just different effect vector, then maybe we could split the
    #  effect into an op + effect to better use fastmode calc?  Or maybe if ordering is right this isn't necessary?
    #Add repcache as in cython version -- saves having to *construct* rep objects all the time... just update
    #coefficients when needed instead?

    #... and we're done!

    target_miss = sum(achieved_sum_of_pathmags) - sum(target_sum_of_pathmags + pathmagnitude_gap)
    if target_miss > 1e-5:
        print("Warning: Achieved sum(path mags) exceeds max by ", target_miss, "!!!")

    #TODO: check that prps are PolynomialReps and not Polynomials -- we may have made this change
    # in fastreplib.pyx but forgot it here.
    return prps, sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)


# foat = first-order always-traversed
def traverse_paths_upto_threshold(oprep_lists, pathmag_threshold, num_elabels, foat_indices_per_op,
                                  fn_visitpath, debug=False):
    """
    Traverse all the paths up to some path-magnitude threshold, calling
    `fn_visitpath` for each one.

    Parameters
    ----------
    oprep_lists : list of lists
        representations for the terms of each layer of the circuit whose
        outcome probability we're computing, including prep and POVM layers.
        `oprep_lists[i]` is a list of the terms available to choose from
        for the i-th circuit layer, ordered by increasing term-magnitude.

    pathmag_threshold : float
        the path-magnitude threshold to use.

    num_elabels : int
        The number of effect labels corresponding whose terms are all
        amassed in the in final `oprep_lists[-1]` list (knowing which
        elements of `oprep_lists[-1]` correspond to which effect isn't
        necessary for this function).

    foat_indices_per_op : list
        A list of lists of integers, such that `foat_indices_per_op[i]`
        is a list of indices into `oprep_lists[-1]` that marks out which
        terms are first-order (Taylor) terms that should therefore always
        be traversed regardless of their term-magnitude (foat = first-order-
        always-traverse).

    fn_visitpath : function
        A function called for each path that is traversed.  Arguments
        are `(term_indices, magnitude, incd)` where `term_indices` is
        an array of integers giving the index into each `oprep_lists[i]`
        list, `magnitude` is the path magnitude, and `incd` is the index
        of the circuit layer that was just incremented (all elements of
        `term_indices` less than this index are guaranteed to be the same
        as they were in the last call to `fn_visitpath`, and this can be
        used for faster path evaluation.

    max_npaths : int, optional
        The maximum number of paths to traverse.  If this is 0, then there
        is no limit.  Otherwise this function will return as soon as
        `max_npaths` paths are traversed.

    debug : bool, optional
        Whether to print additional debug info.

    Returns
    -------
    None
    """  # zot = zero-order-terms
    n = len(oprep_lists)
    nops = [len(oprep_list) for oprep_list in oprep_lists]
    b = [0] * n  # root
    log_thres = _math.log10(pathmag_threshold)

    ##TODO REMOVE
    #if debug:
    #    if debug > 1: print("BEGIN TRAVERSAL")
    #    accepted_bs_and_mags = {}

    def traverse_tree(root, incd, log_thres, current_mag, current_logmag, order, current_nzeros):
        """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
        b = root
        #print("BEGIN: ",root)
        for i in reversed(range(incd, n)):
            if b[i] + 1 == nops[i]: continue
            b[i] += 1

            if order == 0:  # then incd doesn't matter b/c can inc anything to become 1st order
                sub_order = 1 if (i != n - 1 or b[i] >= num_elabels) else 0
            elif order == 1:
                # we started with a first order term where incd was incremented, and now
                # we're incrementing something else
                sub_order = 1 if i == incd else 2  # signifies anything over 1st order where >1 column has be inc'd
            else:
                sub_order = order

            logmag = current_logmag + (oprep_lists[i][b[i]].logmagnitude - oprep_lists[i][b[i] - 1].logmagnitude)
            #print("Trying: ",b)
            if logmag >= log_thres:  # or sub_order == 0:
                numerator = oprep_lists[i][b[i]].magnitude
                denom = oprep_lists[i][b[i] - 1].magnitude
                nzeros = current_nzeros

                if denom == 0:
                    denom = SMALL; nzeros -= 1
                if numerator == 0:
                    numerator = SMALL; nzeros += 1

                mag = current_mag * (numerator / denom)
                actual_mag = mag if (nzeros == 0) else 0.0  # magnitude is actually zero if nzeros > 0

                if fn_visitpath(b, actual_mag, i): return True  # fn_visitpath can signal early return
                if traverse_tree(b, i, log_thres, mag, logmag, sub_order, nzeros):
                    # add any allowed paths beneath this one
                    return True
            elif sub_order <= 1:
                #We've rejected term-index b[i] (in column i) because it's too small - the only reason
                # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
                # iterate through any remaining foat indices for this column (we've accepted all lower
                # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
                # we don't need to traverse down, since we know the path magnitude is already too low.
                orig_bi = b[i]
                for j in foat_indices_per_op[i]:
                    if j >= orig_bi:
                        b[i] = j
                        nzeros = current_nzeros
                        numerator = oprep_lists[i][b[i]].magnitude
                        denom = oprep_lists[i][orig_bi - 1].magnitude
                        if denom == 0: denom = SMALL

                        #if numerator == 0: nzeros += 1  # not needed b/c we just leave numerator = 0
                        # OK if mag == 0 as it's not passed to any recursive calls
                        mag = current_mag * (numerator / denom)
                        actual_mag = mag if (nzeros == 0) else 0.0  # magnitude is actually zero if nzeros > 0

                        if fn_visitpath(b, actual_mag, i): return True

                        if i != n - 1:
                            # if we're not incrementing (from a zero-order term) the final index, then we
                            # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                            orig_bn = b[n - 1]
                            for k in range(1, num_elabels):
                                b[n - 1] = k
                                numerator = oprep_lists[n - 1][b[n - 1]].magnitude
                                denom = oprep_lists[i][orig_bn].magnitude
                                if denom == 0: denom = SMALL
                                # zero if either numerator == 0 or mag == 0 from above.
                                mag2 = mag * (numerator / denom)
                                if fn_visitpath(b, mag2 if (nzeros == 0) else 0.0, n - 1): return True

                            b[n - 1] = orig_bn

                b[i] = orig_bi

            b[i] -= 1  # so we don't have to copy b
        #print("END: ",root)
        return False  # return value == "do we need to terminate traversal immediately?"

    current_mag = 1.0; current_logmag = 0.0
    fn_visitpath(b, current_mag, 0)  # visit root (all 0s) path
    traverse_tree(b, 0, log_thres, current_mag, current_logmag, 0, 0)

    return


# TODO REMOVE: method to traverse paths until the result converges, but I don't think this is well justified
# def traverse_paths_upto_threshold_ex(oprep_lists, high_threshold, low_threshold, num_elabels, foat_indices_per_op,
#                                      fn_visitpath, debug=False):
#     """
#     TODO: docstring
#     """
#     # zot = zero-order-terms
#     n = len(oprep_lists)
#     nops = [len(oprep_list) for oprep_list in oprep_lists]
#     b = [0] * n  # root
#     log_thres_high = _np.log10(high_threshold)  # a previous threshold: we've already visited everything above this
#     log_thres_low = _np.log10(low_threshold)  # visit everything above this threshold
#
#     ##TODO REMOVE
#     #if debug:
#     #    if debug > 1: print("BEGIN TRAVERSAL")
#     #    accepted_bs_and_mags = {}
#
#     def traverse_tree(root, incd, log_thres_high, log_thres_low, current_mag, current_logmag, order):
#         """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
#         b = root
#         #print("BEGIN: ",root)
#         for i in reversed(range(incd, n)):
#             if b[i] + 1 == nops[i]: continue
#             b[i] += 1
#
#             if order == 0:  # then incd doesn't matter b/c can inc anything to become 1st order
#                 sub_order = 1 if (i != n - 1 or b[i] >= num_elabels) else 0
#             elif order == 1:
#                 # we started with a first order term where incd was incremented, and now
#                 # we're incrementing something else
#                 sub_order = 1 if i == incd else 2  # signifies anything over 1st order where >1 column has be inc'd
#             else:
#                 sub_order = order
#
#             logmag = current_logmag + (oprep_lists[i][b[i]].logmagnitude - oprep_lists[i][b[i] - 1].logmagnitude)
#             #print("Trying: ",b)
#             if logmag >= log_thres_low:  # or sub_order == 0:
#                 if oprep_lists[i][b[i] - 1].magnitude == 0:
#                     mag = 0
#                 else:
#                     mag = current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][b[i] - 1].magnitude)
#
#                 if logmag > log_thres_high:
#                     if fn_visitpath(b, mag, i): return True  # fn_visitpath can signal early return
#                 if traverse_tree(b, i, log_thres_high, log_thres_low, mag, logmag, sub_order):
#                     # add any allowed paths beneath this one
#                     return True
#             elif sub_order <= 1 and high_threshold >= 1.0:
#                 #We've rejected term-index b[i] (in column i) because it's too small - the only reason
#                 # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
#                 # iterate through any remaining foat indices for this column (we've accepted all lower
#                 # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
#                 # we don't need to traverse down, since we know the path magnitude is already too low.
#                 orig_bi = b[i]
#                 for j in foat_indices_per_op[i]:
#                     if j >= orig_bi:
#                         b[i] = j
#                         mag = 0 if oprep_lists[i][orig_bi - 1].magnitude == 0 else \
#                             current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][orig_bi - 1].magnitude)
#
#                         if fn_visitpath(b, mag, i): return True
#
#                         if i != n - 1:
#                             # if we're not incrementing (from a zero-order term) the final index, then we
#                             # need to to increment it until we hit num_elabels (*all* zero-th order paths)
#                             orig_bn = b[n - 1]
#                             for k in range(1, num_elabels):
#                                 b[n - 1] = k
#                                 mag2 = mag * (oprep_lists[n - 1][b[n - 1]].magnitude
#                                               / oprep_lists[i][orig_bn].magnitude)
#                                 if fn_visitpath(b, mag2, n - 1): return True
#
#                             b[n - 1] = orig_bn
#
#                 b[i] = orig_bi
#
#             b[i] -= 1  # so we don't have to copy b
#         #print("END: ",root)
#         return False  # return value == "do we need to terminate traversal immediately?"
#
#     current_mag = 1.0; current_logmag = 0.0
#     fn_visitpath(b, current_mag, 0)  # visit root (all 0s) path
#     return traverse_tree(b, 0, log_thres_high, log_thres_low, current_mag, current_logmag, 0)
#     #returns whether fn_visitpath caused us to exit


def pathmagnitude_threshold(oprep_lists, e_indices, num_elabels, target_sum_of_pathmags,
                            foat_indices_per_op=None, initial_threshold=0.1,
                            min_threshold=1e-10, max_npaths=1000000):
    """
    Find the pathmagnitude-threshold needed to achieve some target sum-of-path-magnitudes:
    so that the sum of all the path-magnitudes greater than this threshold achieve the
    target (or get as close as we can).

    Parameters
    ----------
    oprep_lists : list of lists
        representations for the terms of each layer of the circuit whose
        outcome probability we're computing, including prep and POVM layers.
        `oprep_lists[i]` is a list of the terms available to choose from
        for the i-th circuit layer, ordered by increasing term-magnitude.

    e_indices : numpy array
        The effect-vector index for each element of `oprep_lists[-1]`
        (representations for *all* effect vectors exist all together
        in `oprep_lists[-1]`).

    num_elabels : int
        The total number of different effects whose reps appear in
        `oprep_lists[-1]` (also one more than the largest index in
        `e_indices`.

    target_sum_of_pathmags : array
        An array of floats of length `num_elabels` giving the target sum of path
        magnitudes desired for each effect (separately).

    foat_indices_per_op : list
        A list of lists of integers, such that `foat_indices_per_op[i]`
        is a list of indices into `oprep_lists[-1]` that marks out which
        terms are first-order (Taylor) terms that should therefore always
        be traversed regardless of their term-magnitude (foat = first-order-
        always-traverse).

    initial_threshold : float
        The starting pathmagnitude threshold to try (this function uses
        an iterative procedure to find a threshold).

    min_threshold : float
        The smallest threshold allowed.  If this amount is reached, it
        is just returned and searching stops.

    max_npaths : int, optional
        The maximum number of paths allowed per effect.

    Returns
    -------
    threshold : float
        The obtained pathmagnitude threshold.
    npaths : numpy array
        An array of length `num_elabels` giving the number of paths selected
        for each of the effect vectors.
    achieved_sopm : numpy array
        An array of length `num_elabels` giving the achieved sum-of-path-
        magnitudes for each of the effect vectors.
    """
    nIters = 0
    threshold = initial_threshold if (initial_threshold >= 0) else 0.1  # default value
    target_mag = target_sum_of_pathmags
    #print("Target magnitude: ",target_mag)
    threshold_upper_bound = 1.0
    threshold_lower_bound = None
    #db_last_threshold = None #DEBUG TODO REMOVE
    #mag = 0; nPaths = 0

    if foat_indices_per_op is None:
        foat_indices_per_op = [()] * len(oprep_lists)

    # REMOVE comm = mem_limit = None  # TODO: make these arguments later?

    def count_path(b, mg, incd):
        mag[e_indices[b[-1]]] += mg
        nPaths[e_indices[b[-1]]] += 1

        # REMOVE?
        # #Instead of magnitude, accumulate actual current path contribution that we can test for convergence
        # factors = [oprep_lists[i][factorInd] for i, factorInd in enumerate(b)]
        # res = _np.product([f.evaluated_coeff for f in factors])
        # pLeft = _unitary_sim_pre(factors, comm, mem_limit)
        # pRight = _unitary_sim_post(factors, comm, mem_limit)
        # res *= (pLeft * pRight)
        #
        # final_factor_indx = b[-1]
        # Ei = e_indices[final_factor_indx]  # final "factor" index == E-vector index
        # integrals[Ei] += res

        return (nPaths[e_indices[b[-1]]] == max_npaths)  # trigger immediate return if hit max_npaths

    while nIters < 100:  # TODO: allow setting max_nIters as an arg?
        mag = _np.zeros(num_elabels, 'd')
        nPaths = _np.zeros(num_elabels, int)

        traverse_paths_upto_threshold(oprep_lists, threshold, num_elabels,
                                      foat_indices_per_op, count_path)  # sets mag and nPaths
        assert(max_npaths == 0 or _np.all(nPaths <= max_npaths)), "MAX PATHS EXCEEDED! (%s)" % nPaths

        if _np.all(mag >= target_mag) or _np.any(nPaths >= max_npaths):  # try larger threshold
            threshold_lower_bound = threshold
            if threshold_upper_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold *= 2
        else:  # try smaller threshold
            threshold_upper_bound = threshold
            if threshold_lower_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold /= 2

        if threshold_upper_bound is not None and threshold_lower_bound is not None and \
           (threshold_upper_bound - threshold_lower_bound) / threshold_upper_bound < 1e-3:
            #print("Converged after %d iters!" % nIters)
            break
        if threshold_upper_bound < min_threshold:  # could also just set min_threshold to be the lower bound initially?
            threshold_upper_bound = threshold_lower_bound = min_threshold
            break

        nIters += 1

    #Run path traversal once more to count final number of paths

    def count_path_nomax(b, mg, incd):
        # never returns True - we want to check *threshold* alone selects correct # of paths
        mag[e_indices[b[-1]]] += mg
        nPaths[e_indices[b[-1]]] += 1

    mag = _np.zeros(num_elabels, 'd')
    # integrals = _np.zeros(num_elabels, 'd') REMOVE
    nPaths = _np.zeros(num_elabels, int)
    traverse_paths_upto_threshold(oprep_lists, threshold_lower_bound, num_elabels,
                                  foat_indices_per_op, count_path_nomax)  # sets mag and nPaths

    #TODO REMOVE - idea of truncating based on convergence of sum seems flawed - can't detect long tails
    # last_threshold = 1e10  # something huge
    # threshold = initial_threshold  # needs to be < 1
    # converged = False
    #
    # while not converged:
    #     converged = traverse_paths_upto_threshold_ex(oprep_lists, last_threshold, threshold,
    #                                                  num_elabels, foat_indices_per_op, count_path)
    #     last_threshold = threshold
    #     threshold /= 2

    return threshold_lower_bound, nPaths, mag


def _unitary_sim_pre(complete_factors, comm, mem_limit):
    rhoVec = complete_factors[0].pre_state  # a prep representation
    for f in complete_factors[0].pre_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.pre_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].pre_ops:
        rhoVec = f.acton(rhoVec)

    EVec = complete_factors[-1].post_effect
    return EVec.amplitude(rhoVec)


def _unitary_sim_post(complete_factors, comm, mem_limit):
    rhoVec = complete_factors[0].post_state  # a prep representation
    for f in complete_factors[0].post_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.post_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].post_ops:
        rhoVec = f.acton(rhoVec)
    EVec = complete_factors[-1].pre_effect
    return _np.conjugate(EVec.amplitude(rhoVec))  # conjugate for same reason as above
