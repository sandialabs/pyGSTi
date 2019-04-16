"""Defines Python-version calculation "representation" objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import sys
import time as _time
import numpy as _np
import scipy.sparse as _sps
import itertools as _itertools
import functools as _functools

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import matrixtools as _mt
from ..tools import listtools as _lt

from scipy.sparse.linalg import LinearOperator

# DEBUG!!!
DEBUG_FCOUNT = 0


class DMStateRep(object):
    def __init__(self, data):
        self.data = _np.asarray(data, 'd')

    def copy_from(self, other):
        self.data = other.data.copy()

    def todense(self):
        return self.data

    @property
    def dim(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)


class DMEffectRep(object):
    def __init__(self, dim):
        self.dim = dim

    def probability(self, state):
        raise NotImplementedError()


class DMEffectRep_Dense(DMEffectRep):
    def __init__(self, data):
        self.data = _np.array(data, 'd')
        super(DMEffectRep_Dense, self).__init__(len(self.data))

    def probability(self, state):
        # can assume state is a DMStateRep
        return _np.dot(self.data, state.data)  # not vdot b/c *real* data


class DMEffectRep_TensorProd(DMEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim  # Unused
        super(DMEffectRep_TensorProd, self).__init__(dim)

    def todense(self, outvec):
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
        Edense = self.todense(scratch)
        return _np.dot(Edense, state.data)  # not vdot b/c data is *real*


class DMEffectRep_Computational(DMEffectRep):
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

        self.nfactors = len(zvals)  # (or nQubits)
        self.abs_elval = 1 / (_np.sqrt(2)**self.nfactors)

        super(DMEffectRep_Computational, self).__init__(dim)

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

    def todense(self, outvec, trust_outvec_sparsity=False):
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
        Edense = self.todense(scratch)
        return _np.dot(Edense, state.data)  # not vdot b/c data is *real*


class DMEffectRep_Errgen(DMEffectRep):  # TODO!! Need to make SV version
    def __init__(self, errgen_oprep, effect_rep, errgen_id):
        dim = effect_rep.dim
        self.errgen_rep = errgen_oprep
        self.effect_rep = effect_rep
        self.errgen_id = errgen_id
        super(DMEffectRep_Errgen, self).__init__(dim)

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
            return self.acton(in_state).todense()

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = DMStateRep(_np.ascontiguousarray(v, 'd'))
            return self.adjoint_acton(in_state).todense()
        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?


class DMOpRep_Dense(DMOpRep):
    def __init__(self, data):
        self.data = data
        super(DMOpRep_Dense, self).__init__(self.data.shape[0])

    def acton(self, state):
        return DMStateRep(_np.dot(self.data, state.data))

    def adjoint_acton(self, state):
        return DMStateRep(_np.dot(self.data.T, state.data))  # no conjugate b/c *real* data

    def __str__(self):
        return "DMOpRep_Dense:\n" + str(self.data)


class DMOpRep_Embedded(DMOpRep):
    def __init__(self, embedded_op, numBasisEls, actionInds,
                 blocksizes, embedded_dim, nComponentsInActiveBlock,
                 iActiveBlock, nBlocks, dim):

        self.embedded_op = embedded_op
        self.numBasisEls = numBasisEls
        self.actionInds = actionInds
        self.blocksizes = blocksizes

        numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds: numBasisEls_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in numBasisEls_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(numBasisEls[1:]))))), _np.int64)
        self.basisInds_action = [list(range(numBasisEls[i])) for i in actionInds]

        self.embeddedDim = embedded_dim
        self.iActiveBlock = iActiveBlock
        self.nBlocks = nBlocks
        self.offset = sum(blocksizes[0:iActiveBlock])
        super(DMOpRep_Embedded, self).__init__(dim)

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.iActiveBlock:
                output_state.data[offset:offset + blockSize] = state.data[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = DMStateRep(_np.zeros(state.data.shape, 'd'))
        offset = self.offset  # if relToBlock else self.offset (relToBlock == False here)

        #print("DB REPLIB ACTON: ",self.basisInds_noop_blankaction)
        #print("DB REPLIB ACTON: ",self.basisInds_action)
        #print("DB REPLIB ACTON: ",self.multipliers)
        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.actionInds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep(state.data[inds])
            embedded_outstate = self.embedded_op.acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = DMStateRep(_np.zeros(state.data.shape, 'd'))
        offset = self.offset  # if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.actionInds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep(state.data[inds])
            embedded_outstate = self.embedded_op.adjoint_acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class DMOpRep_Composed(DMOpRep):
    def __init__(self, factor_op_reps, dim):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorops = factor_op_reps
        super(DMOpRep_Composed, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorops:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factorops):
            state = gate.adjoint_acton(state)
        return state


class DMOpRep_Sum(DMOpRep):
    def __init__(self, factor_reps, dim):
        #assert(len(factor_reps) > 0), "Summed gates must contain at least one factor gate!"
        self.factors = factor_reps
        super(DMOpRep_Sum, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = DMStateRep(_np.zeros(state.data.shape, 'd'))
        for f in self.factors:
            output_state.data += f.acton(state).data
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = DMStateRep(_np.zeros(state.data.shape, 'd'))
        for f in self.factors:
            output_state.data += f.adjoint_acton(state).data
        return output_state


class DMOpRep_Exponentiated(DMOpRep):
    def __init__(self, exponentiated_op_rep, power, dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(DMOpRep_Exponentiated, self).__init__(dim)

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


class DMOpRep_Lindblad(DMOpRep):
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
        super(DMOpRep_Lindblad, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        if self.unitary_postfactor is not None:
            statedata = self.unitary_postfactor.dot(state.data)
        else:
            statedata = state.data

        tol = 1e-16  # 2^-53 (=Scipy default) -- TODO: make into an arg?
        A = self.errgen_rep.aslinearoperator()  # ~= a sparse matrix for call below
        statedata = _mt._custom_expm_multiply_simple_core(
            A, statedata, self.mu, self.m_star, self.s, tol, self.eta)
        return DMStateRep(statedata)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for sparse Lindblad LinearOperator Reps yet.")


class DMOpRep_Sparse(DMOpRep):
    def __init__(self, A_data, A_indices, A_indptr):
        dim = len(A_indptr) - 1
        self.A = _sps.csr_matrix((A_data, A_indices, A_indptr), shape=(dim, dim))
        super(DMOpRep_Sparse, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        return DMStateRep(self.A.dot(state.data))

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        Aadj = self.A.conjugate(copy=True).transpose()
        return DMStateRep(Aadj.dot(state.data))


# State vector (SV) propagation wrapper classes
class SVStateRep(object):
    def __init__(self, data):
        self.data = _np.asarray(data, complex)

    def copy_from(self, other):
        self.data = other.data.copy()

    @property
    def dim(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)


class SVEffectRep(object):
    def __init__(self, dim):
        self.dim = dim

    def probability(self, state):
        return abs(self.amplitude(state))**2

    def amplitude(self, state):
        raise NotImplementedError()


class SVEffectRep_Dense(SVEffectRep):
    def __init__(self, data):
        self.data = _np.array(data, complex)
        super(SVEffectRep_Dense, self).__init__(len(self.data))

    def amplitude(self, state):
        # can assume state is a SVStateRep
        return _np.vdot(self.data, state.data)  # (or just 'dot')


class SVEffectRep_TensorProd(SVEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim  # Unused
        super(SVEffectRep_TensorProd, self).__init__(dim)

    def todense(self, outvec):
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
        Edense = self.todense(scratch)
        return _np.vdot(Edense, state.data)


class SVEffectRep_Computational(SVEffectRep):
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
        for k, v in enumerate(zvals):
            assert(v in (0, 1)), "zvals must contain only 0s and 1s"
            self.nonzero_index += base * v
            base //= 2  # or right shift?
        super(SVEffectRep_Computational, self).__init__(dim)

    def todense(self, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if not trust_outvec_sparsity:
            outvec[:] = 0  # reset everything to zero
        outvec[self.nonzero_index] = 1.0
        return outvec

    def amplitude(self, state):  # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.todense(scratch)
        return _np.vdot(Edense, state.data)


class SVOpRep(object):
    def __init__(self, dim):
        self.dim = dim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()


class SVOpRep_Dense(SVOpRep):
    def __init__(self, data):
        self.data = data
        super(SVOpRep_Dense, self).__init__(self.data.shape[0])

    def acton(self, state):
        return SVStateRep(_np.dot(self.data, state.data))

    def adjoint_acton(self, state):
        return SVStateRep(_np.dot(_np.conjugate(self.data.T), state.data))

    def __str__(self):
        return "SVOpRep_Dense:\n" + str(self.data)


class SVOpRep_Embedded(SVOpRep):
    # exactly the same as DM case
    def __init__(self, embedded_op, numBasisEls, actionInds,
                 blocksizes, embedded_dim, nComponentsInActiveBlock,
                 iActiveBlock, nBlocks, dim):

        self.embedded_op = embedded_op
        self.numBasisEls = numBasisEls
        self.actionInds = actionInds
        self.blocksizes = blocksizes

        numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds: numBasisEls_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in numBasisEls_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(numBasisEls[1:]))))), _np.int64)
        self.basisInds_action = [list(range(numBasisEls[i])) for i in actionInds]

        self.embeddedDim = embedded_dim
        self.nComponents = nComponentsInActiveBlock
        self.iActiveBlock = iActiveBlock
        self.nBlocks = nBlocks
        self.offset = sum(blocksizes[0:iActiveBlock])
        super(SVOpRep_Embedded, self).__init__(dim)

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.iActiveBlock:
                output_state.data[offset:offset + blockSize] = state.data[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = SVStateRep(_np.zeros(state.data.shape, complex))
        offset = self.offset  # if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.actionInds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep(state.data[inds])
            embedded_outstate = self.embedded_op.acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = SVStateRep(_np.zeros(state.data.shape, complex))
        offset = self.offset  # if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.actionInds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep(state.data[inds])
            embedded_outstate = self.embedded_op.adjoint_acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class SVOpRep_Composed(SVOpRep):
    # exactly the same as DM case
    def __init__(self, factor_op_reps, dim):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorsgates = factor_op_reps
        super(SVOpRep_Composed, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorops:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factorops):
            state = gate.adjoint_acton(state)
        return state


class SVOpRep_Sum(SVOpRep):
    # exactly the same as DM case
    def __init__(self, factor_reps, dim):
        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factors = factor_reps
        super(SVOpRep_Sum, self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = SVStateRep(_np.zeros(state.data.shape, complex))
        for f in self.factors:
            output_state.data += f.acton(state).data
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = SVStateRep(_np.zeros(state.data.shape, complex))
        for f in self.factors:
            output_state.data += f.adjoint_acton(state).data
        return output_state


class SVOpRep_Exponentiated(SVOpRep):
    def __init__(self, exponentiated_op_rep, power, dim):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(SVOpRep_Exponentiated, self).__init__(dim)

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
        from .stabilizer import StabilizerFrame as _StabilizerFrame
        self.sframe = _StabilizerFrame(smatrix, pvectors, amps)
        # just rely on StabilizerFrame class to do all the heavy lifting...

    def copy(self):
        cpy = SBStateRep(_np.zeros((0, 0), _np.int64), None, None)  # makes a dummy cpy.sframe
        cpy.sframe = self.sframe.copy()  # a legit copy *with* qubit filers copied too
        return cpy

    @property
    def nqubits(self):
        return self.sframe.n

    def __str__(self):
        return "SBStateRep:\n" + str(self.sframe)


class SBEffectRep(object):
    def __init__(self, zvals):
        self.zvals = zvals

    @property
    def nqubits(self):
        return len(self.zvals)

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


class SBOpRep_Embedded(SBOpRep):
    def __init__(self, embedded_op, n, qubits):
        self.embedded_op = embedded_op
        self.qubit_indices = qubits
        super(SBOpRep_Embedded, self).__init__(n)

    def acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubit_indices)
        outstate = self.embedded_op.acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate

    def adjoint_acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubit_indices)
        outstate = self.embedded_op.adjoint_acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate


class SBOpRep_Composed(SBOpRep):
    # exactly the same as DM case except .dim -> .n
    def __init__(self, factor_op_reps, n):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorops = factor_op_reps
        super(SBOpRep_Composed, self).__init__(n)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorops:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factorops):
            state = gate.adjoint_acton(state)
        return state


class SBOpRep_Sum(SBOpRep):
    # exactly the same as DM case except .dim -> .n
    def __init__(self, factor_reps, n):
        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factors = factor_reps
        super(SBOpRep_Sum, self).__init__(n)

    def acton(self, state):
        """ Act this gate map on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()


class SBOpRep_Exponentiated(SBOpRep):
    def __init__(self, exponentiated_op_rep, power, n):
        self.exponentiated_op = exponentiated_op_rep
        self.power = power
        super(SBOpRep_Exponentiated, self).__init__(n)

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


class SBOpRep_Clifford(SBOpRep):
    def __init__(self, smatrix, svector, smatrix_inv, svector_inv, unitary):
        self.smatrix = smatrix
        self.svector = svector
        self.smatrix_inv = smatrix_inv
        self.svector_inv = svector_inv
        self.unitary = unitary
        super(SBOpRep_Clifford, self).__init__(smatrix.shape[0] // 2)

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
class PolyRep(dict):
    """
    Representation class for a polynomial.

    This is similar to a full Polynomial
    dictionary, but lacks some functionality and is optimized for computation
    speed.  In particular, the keys of this dict are not tuples of variable
    indices (as in Polynomial) but simple integers encoded from such tuples.
    To perform this mapping, one must specify a maximum order and number of
    variables.
    """

    def __init__(self, int_coeffs, max_order, max_num_vars, vindices_per_int):
        """
        Create a new PolyRep object.

        Parameters
        ----------
        int_coeffs : dict
            A dictionary of coefficients whose keys are already-encoded
            integers corresponding to variable-index-tuples (i.e poly
            terms).

        max_order : int
            The maximum order (exponent) allowed for any single variable
            in each monomial term.

        max_num_vars : int
            The maximum number of variables allowed.  For example, if
            set to 2, then only "x0" and "x1" are allowed to appear
            in terms.
        """

        self.max_order = max_order
        self.max_num_vars = max_num_vars
        self.vindices_per_int = vindices_per_int

        super(PolyRep, self).__init__()
        if int_coeffs is not None:
            self.update(int_coeffs)

    @property
    def coeffs(self):  # so we can convert back to python Polys
        """ The coefficient dictionary (with encoded integer keys) """
        return dict(self)  # for compatibility w/C case which can't derive from dict...

    def set_maximums(self, max_num_vars=None, max_order=None):
        """
        Alter the maximum order and number of variables (and hence the
        tuple-to-int mapping) for this polynomial representation.

        Parameters
        ----------
        max_num_vars : int
            The maximum number of variables allowed.

        max_order : int
            The maximum order (exponent) allowed for any single variable
            in each monomial term.

        Returns
        -------
        None
        """
        coeffs = {self._int_to_vinds(k): v for k, v in self.items()}
        if max_num_vars is not None: self.max_num_vars = max_num_vars
        if max_order is not None: self.max_order = max_order
        int_coeffs = {self._vinds_to_int(k): v for k, v in coeffs.items()}
        self.clear()
        self.update(int_coeffs)

    def _vinds_to_int(self, vinds):
        """ Maps tuple of variable indices to encoded int """
        ints_in_key = int(_np.ceil(len(vinds) / self.vindices_per_int))
        #OLD (before multi-int keys): assert(len(vinds) <= self.max_order), "max_order (%d) is too low!" % self.max_order

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

    def deriv(self, wrtParam):
        """
        Take the derivative of this polynomial representation with respect to
        the single variable `wrtParam`.

        Parameters
        ----------
        wrtParam : int
            The variable index to differentiate with respect to (can be
            0 to the `max_num_vars-1` supplied to `__init__`.

        Returns
        -------
        PolyRep
        """
        dcoeffs = {}
        for i, coeff in self.items():
            ivar = self._int_to_vinds(i)
            cnt = float(ivar.count(wrtParam))
            if cnt > 0:
                l = list(ivar)
                del l[ivar.index(wrtParam)]
                dcoeffs[tuple(l)] = cnt * coeff
        int_dcoeffs = {self._vinds_to_int(k): v for k, v in dcoeffs.items()}
        return PolyRep(int_dcoeffs, self.max_order, self.max_num_vars, self.vindices_per_int)

    def evaluate(self, variable_values):
        """
        Evaluate this polynomial at the given variable values.

        Parameters
        ----------
        variable_values : iterable
            The values each variable will be evaluated at.  Must have
            length at least equal to the number of variables present
            in this `PolyRep`.

        Returns
        -------
        float or complex
        """
        #FUTURE and make this function smarter (Russian peasant)?
        ret = 0
        for i, coeff in self.items():
            ivar = self._int_to_vinds(i)
            ret += coeff * _np.product([variable_values[i] for i in ivar])
        return ret

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
            A 1D array of coefficients; can have either real
            or complex data type.
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

    def copy(self):
        """
        Make a copy of this polynomial representation.

        Returns
        -------
        PolyRep
        """
        cpy = PolyRep(None, self.max_order, self.max_num_vars, self.vindices_per_int)
        cpy.update(self)  # constructor expects dict w/var-index keys, not ints like self has
        return cpy

    def map_indices_inplace(self, mapfn):
        """
        Map the variable indices in this `PolyRep`.
        This allows one to change the "labels" of the variables.

        Parameters
        ----------
        mapfn : function
            A single-argument function that maps old variable-index tuples
            to new ones.  E.g. `mapfn` might map `(0,1)` to `(10,11)` if
            we were increasing each variable index by 10.

        Returns
        -------
        None
        """
        new_items = {self._vinds_to_int(mapfn(self._int_to_vinds(k))): v
                     for k, v in self.items()}
        self.clear()
        self.update(new_items)

    def mult(self, x):
        """
        Returns `self * x` where `x` is another polynomial representation.

        Parameters
        ----------
        x : PolyRep

        Returns
        -------
        PolyRep
        """
        assert(self.max_order == x.max_order and self.max_num_vars == x.max_num_vars)
        # assume same *fixed* max_order, even during mult
        newpoly = PolyRep(None, self.max_order, self.max_num_vars, self.vindices_per_int)
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
        newpoly = self.copy()
        for k in self:
            self[k] *= x

    def scalar_mult(self, x):
        """
        Returns `self * x` where `x` is a scalar.

        Parameters
        ----------
        x : float or complex

        Returns
        -------
        PolyRep
        """
        # assume a scalar that can multiply values
        newpoly = self.copy()
        newpoly.scale(x)
        return newpoly

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
        return "PolyRep[ " + str(self) + " ]"

    def __add__(self, x):
        newpoly = self.copy()
        if isinstance(x, PolyRep):
            assert(self.max_order == x.max_order and self.max_num_vars == x.max_num_vars)
            for k, v in x.items():
                if k in newpoly: newpoly[k] += v
                else: newpoly[k] = v
        else:  # assume a scalar that can be added to values
            for k in newpoly:
                newpoly[k] += x
        return newpoly

    def __mul__(self, x):
        if isinstance(x, PolyRep):
            return self.mult_poly(x)
        else:  # assume a scalar that can multiply values
            return self.mult_scalar(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self, n):
        ret = PolyRep({0: 1.0}, self.max_order, self.max_num_vars, self.vindices_per_int)
        cur = self
        for i in range(int(_np.floor(_np.log2(n))) + 1):
            rem = n % 2  # gets least significant bit (i-th) of n
            if rem == 1: ret *= cur  # add current power of x (2^i) if needed
            cur = cur * cur  # current power *= 2
            n //= 2  # shift bits of n right
        return ret

    def __copy__(self):
        return self.copy()

    def debug_report(self):
        actual_max_order = max([len(self._int_to_vinds(k)) for k in self.keys()])
        return "PolyRep w/max_vars=%d and max_order=%d: nterms=%d, actual max-order=%d" % \
            (self.max_num_vars, self.max_order, len(self), actual_max_order)

    def degree(self):
        return max([len(self._int_to_vinds(k)) for k in self.keys()])


class SVTermRep(object):
    # just a container for other reps (polys, states, effects, and gates)
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


class SBTermRep(object):
    # exactly the same as SVTermRep
    # just a container for other reps (polys, states, effects, and gates)
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


# No need to create separate classes for floating-pt (vs. polynomial) coeff in Python (no types!)
SVTermDirectRep = SVTermRep
SBTermDirectRep = SBTermRep

## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, operationreps):
    ret = staterep
    for oprep in operationreps:
        ret = oprep.acton(ret)
    return ret


def DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, scratch=None):  # TODO remove scratch

    cacheSize = evalTree.cache_size()
    rhoVec, EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    ret = _np.empty((len(evalTree), len(elabels)), 'd')

    #Get rho & rhoCache
    rho = rhoVec.torep('prep')
    rho_cache = [None] * cacheSize  # so we can store (s,p) tuples in cache

    #Get operationreps and ereps now so we don't make unnecessary .torep() calls
    operationreps = {gl: calc.sos.get_operation(gl).torep() for gl in evalTree.opLabels}
    ereps = [E.torep('effect') for E in EVecs]

    #REMOVE?? - want some way to speed tensorprod effect actions...
    #if self.evotype in ("statevec", "densitymx"):
    #    Escratch = _np.empty(self.dim,typ) # memory for E.todense() if it wants it

    #comm is currently ignored
    #TODO: if evalTree is split, distribute among processors
    for i in evalTree.get_evaluation_order():
        iStart, remainder, iCache = evalTree[i]
        if iStart is None: init_state = rho
        else: init_state = rho_cache[iStart]  # [:,None]

        #OLD final_state = self.propagate_state(init_state, remainder)
        final_state = propagate_staterep(init_state, [operationreps[gl] for gl in remainder])
        if iCache is not None: rho_cache[iCache] = final_state  # [:,0] #store this state in the cache

        #HERE - current_errgen_name check?
        for j, erep in enumerate(ereps):
            ret[i, j] = erep.probability(final_state)  # outcome probability

    #print("DEBUG TIME: pr_cache(dim=%d, cachesize=%d) in %gs" % (self.dim, cacheSize,_time.time()-tStart)) #DEBUG

    return ret


def DM_compute_dpr_cache(calc, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):

    eps = 1e-7  # hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(calc.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
    nDerivCols = len(param_indices)  # *all*, not just locally computed ones

    rhoVec, EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = _np.empty((len(evalTree), len(elabels)), 'd')
    dpr_cache = _np.zeros((len(evalTree), len(elabels), nDerivCols), 'd')

    #Get (extension-type) representation objects
    rhorep = calc.sos.get_prep(rholabel).torep('prep')
    ereps = [calc.sos.get_effect(el).torep('effect') for el in elabels]
    operation_lookup = {lbl: i for i, lbl in enumerate(evalTree.opLabels)}  # operation labels -> ints for faster lookup
    operationreps = {i: calc.sos.get_operation(lbl).torep() for lbl, i in operation_lookup.items()}
    cacheSize = evalTree.cache_size()

    # create rho_cache (or use scratch)
    if scratch is None:
        rho_cache = [None] * cacheSize  # so we can store (s,p) tuples in cache
    else:
        assert(len(scratch) == cacheSize)
        rho_cache = scratch

    pCache = DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, rho_cache)  # here scratch is used...

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec)
            dpr_cache[:, :, iFinal] = (DM_compute_pr_cache(
                calc, rholabel, elabels, evalTree, subComm, rho_cache) - pCache) / eps
    calc.from_vector(orig_vec)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, dpr_cache, [], axes=2, comm=comm)

    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (self.Np, self.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG

    return dpr_cache


def SV_prs_as_polys(calc, rholabel, elabels, circuit, comm=None, memLimit=None, fastmode=True):
    return _prs_as_polys(calc, rholabel, elabels, circuit, comm, memLimit, fastmode)


def SB_prs_as_polys(calc, rholabel, elabels, circuit, comm=None, memLimit=None, fastmode=True):
    return _prs_as_polys(calc, rholabel, elabels, circuit, comm, memLimit, fastmode)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_polys(calc, rholabel, elabels, circuit, comm=None, memLimit=None, fastmode=True):
    """
    Computes polynomials of the probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    calc : TermForwardSimulator
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

    memLimit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    Returns
    -------
    list
        A list of PolyRep objects, one per element of `elabels`.
    """
    #print("PRS_AS_POLY circuit = ",circuit)
    #print("DB: prs_as_polys(",spamTuple,circuit,calc.max_order,")")

    #NOTE for FUTURE: to adapt this to work with numerical rather than polynomial coeffs:
    # mpo = mpv = None # (these shouldn't matter)
    # use get_direct_order_terms(order, order_base) w/order_base=0.1(?) instead of get_taylor_order_terms??
    # below: replace prps with: prs = _np.zeros(len(elabels),complex)  # an array in "bulk" mode
    #  use *= or * instead of .mult( and .scale(
    #  e.g. res = _np.product([f.coeff for f in factors])
    #       res *= (pLeft * pRight)
    # - add assert(_np.linalg.norm(_np.imag(prs)) < 1e-6) at end and return _np.real(prs)

    mpv = calc.Np  # max_poly_vars
    mpo = calc.max_order * 2  # max_poly_order

    # Construct dict of gate term reps
    distinct_gateLabels = sorted(set(circuit))
    op_term_reps = {glbl: [[t.torep(mpo, mpv, "gate") for t in calc.sos.get_operation(glbl).get_taylor_order_terms(order)]
                           for order in range(calc.max_order + 1)]
                    for glbl in distinct_gateLabels}

    #Similar with rho_terms and E_terms, but lists
    rho_term_reps = [[t.torep(mpo, mpv, "prep") for t in calc.sos.get_prep(rholabel).get_taylor_order_terms(order)]
                     for order in range(calc.max_order + 1)]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order + 1):
        cur_term_reps = []  # the term reps for *all* the effect vectors
        cur_indices = []  # the Evec-index corresponding to each term rep
        for i, elbl in enumerate(elabels):
            term_reps = [t.torep(mpo, mpv, "effect") for t in calc.sos.get_effect(elbl).get_taylor_order_terms(order)]
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
    db_part_cnt = 0
    db_factor_cnt = 0
    #print("DB: pr_as_poly for ",str(tuple(map(str,circuit))), " max_order=",calc.max_order)

    prps = [None] * len(elabels)  # an array in "bulk" mode? or Polynomial in "symbolic" mode?
    for order in range(calc.max_order + 1):
        #print("DB: pr_as_poly order=",order)
        db_npartitions = 0
        for p in _lt.partition_into(order, len(circuit) + 2):  # +2 for SPAM bookends
            #factor_lists = [ calc.sos.get_operation(glbl).get_order_terms(pi) for glbl,pi in zip(circuit,p) ]
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

                    for f in factors[-1].post_ops:
                        rhoVecL = f.acton(rhoVecL)
                    E = factors[-1].post_effect  # effect representation
                    pLeft = E.amplitude(rhoVecL)

                    #Same for pre_ops and rhoVecR
                    for f in factors[-1].pre_ops:
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
                    #print("DB PYHON: prps[%d] = " % Ei, prps[Ei])

            else:  # non-fast mode
                last_index = len(factor_lists) - 1
                for fi in _itertools.product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(fi)]
                    res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                    pLeft = _unitary_sim_pre(factors, comm, memLimit)
                    pRight = _unitary_sim_post(factors, comm, memLimit)
                    # if not self.unitary_evolution else 1.0
                    res.scale((pLeft * pRight))
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx]  # final "factor" index == E-vector index
                    #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
                    if prps[Ei] is None: prps[Ei] = res
                    else: prps[Ei] += res  # add_inplace?
                    #print("DB running prps[",Ei,"] =",prps[Ei])

            #DEBUG!!!
            db_nfactors = [len(l) for l in factor_lists]
            db_totfactors = _np.product(db_nfactors)
            db_factor_cnt += db_totfactors
            DEBUG_FCOUNT += db_totfactors
            db_part_cnt += 1
            #print("DB: pr_as_poly   partition=",p,"(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

    #print("DONE -> FCOUNT=",DEBUG_FCOUNT)
    return prps  # can be a list of polys


def SV_prs_directly(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, wtTol=0.0, resetTermWeights=True, debug=None):
    #return _prs_directly(calc, rholabel, elabels, circuit, comm, memLimit, fastmode)
    raise NotImplementedError("No direct mode yet")


def SB_prs_directly(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, wtTol=0.0, resetTermWeights=True, debug=None):
    #return _prs_directly(calc, rholabel, elabels, circuit, comm, memLimit, fastmode)
    raise NotImplementedError("No direct mode yet")


def SV_prs_as_pruned_polys(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, pathmagnitude_gap=0.0, min_term_mag=0.01,
                           current_threshold=None):
    return _prs_as_pruned_polys(calc, rholabel, elabels, circuit, repcache, comm, memLimit, fastmode, pathmagnitude_gap, min_term_mag,
                                current_threshold)


def SB_prs_as_pruned_polys(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, pathmagnitude_gap=0.0, min_term_mag=0.01):
    return _prs_as_pruned_polys(calc, rholabel, elabels, circuit, repcache, comm, memLimit, fastmode, pathmagnitude_gap, min_term_mag)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_pruned_polys(calc, rholabel, elabels, circuit, repcache, comm=None, memLimit=None, fastmode=True, pathmagnitude_gap=0.0, min_term_mag=0.01,
                         current_threshold=None):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    calc : TermForwardSimulator
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

    memLimit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    TODO: docstring - additional args and now return polys again

    Returns
    -------
    numpy.ndarray
        one per element of `elabels`.
    """
    #t0 = _time.time()

    # Construct dict of gate term reps
    mpv = calc.Np  # max_poly_vars
    mpo = 1000  # PLATFORM_BITS / _np.log2(mpv) #max_poly_order allowed for our integer storage
    distinct_gateLabels = sorted(set(circuit))

    #TODO REMOVE
    #print("DB: _prs_as_pruned_polys: ", circuit, "Distinct = ",distinct_gateLabels)
    #op_term_reps = { glbl: [ t.torep(mpo,mpv,"gate") for t in calc.sos.get_operation(glbl).get_highmagnitude_terms(min_term_mag, max_taylor_order=calc.max_order)]
    #                 for glbl in distinct_gateLabels }
    #op_num_foat = { glbl: calc.sos.get_operation(glbl).get_num_firstorder_terms() for glbl in distinct_gateLabels }

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = calc.sos.get_operation(glbl).get_highmagnitude_terms(
                min_term_mag, max_taylor_order=calc.max_order)
            repcache[glbl] = ([t.torep(mpo, mpv, "gate") for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    #Similar with rho_terms and E_terms, but lists
    #rho_term_reps = [ t.torep(mpo,mpv,"prep") for t in calc.sos.get_prep(rholabel).get_highmagnitude_terms(min_term_mag, max_taylor_order=calc.max_order) ]
    #rho_num_foat = calc.sos.get_prep(rholabel).get_num_firstorder_terms() # TODO REMOVE
    if rholabel not in repcache:
        hmterms, foat_indices = calc.sos.get_prep(rholabel).get_highmagnitude_terms(
            min_term_mag, max_taylor_order=calc.max_order)
        repcache[rholabel] = ([t.torep(mpo, mpv, "prep") for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = calc.sos.get_effect(elbl).get_highmagnitude_terms(
                min_term_mag, max_taylor_order=calc.max_order)
            E_term_indices_and_reps.extend(
                [(i, t.torep(mpo, mpv, "effect"), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] == True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)

    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    prps = [None] * len(elabels)

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    factor_list_lens = list(map(len, factor_lists))
    last_index = len(factor_lists) - 1

    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [calc.sos.get_prep(rholabel)] + [calc.sos.get_operation(glbl) for glbl in circuit]
    max_sum_of_pathmags = _np.product([op.get_total_term_magnitude() for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * calc.sos.get_effect(elbl).get_total_term_magnitude() for elbl in elabels], 'd')
    target_sum_of_pathmags = max_sum_of_pathmags - pathmagnitude_gap
    threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
        factor_lists, E_indices, len(elabels), target_sum_of_pathmags, len(elabels), foat_indices_per_op,
        initial_threshold=current_threshold, min_threshold=pathmagnitude_gap / 100.0)
    # above takes an array of target pathmags and gives a single threshold that works for all of them (all E-indices)

    if current_threshold >= 0 and threshold >= current_threshold:  # then just keep existing (cached) polys
        return None, sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)

    #print("T1 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    if fastmode:
        leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
        rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
        coeffSaved = [None] * (len(factor_lists) - 1)

        def add_path(b, mag, incd):
            """ Relies on the fact that paths are iterated over in lexographic order, and `incd`
                tells us which index was just incremented (all indices less than this one are
                the *same* as the last call). """
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no ability to cache?)
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
            for f in factors[-1].post_ops:
                rhoVecL = f.acton(rhoVecL)
            E = factors[-1].post_effect  # effect representation
            pLeft = E.amplitude(rhoVecL)

            #Same for pre_ops and rhoVecR
            for f in factors[-1].pre_ops:
                rhoVecR = f.acton(rhoVecR)
            E = factors[-1].pre_effect
            pRight = _np.conjugate(E.amplitude(rhoVecR))

            res = coeff.mult(factors[-1].coeff)
            res.scale((pLeft * pRight))
            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index

            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei] += res  # could add_inplace?

    else:
        def add_path(b, mag, incd):
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no ability to cache?)
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
            pLeft = _unitary_sim_pre(factors, comm, memLimit)
            pRight = _unitary_sim_post(factors, comm, memLimit)
            res.scale((pLeft * pRight))

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei] += res  # add_inplace?
            #print("DB running prps[",Ei,"] =",prps[Ei])

    traverse_paths_upto_threshold(factor_lists, threshold, len(
        elabels), foat_indices_per_op, add_path)  # sets mag and nPaths
    #print("T2 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    #DEBUG
    #print("---------------------------")
    #print("Path threshold = ",threshold, " max=",max_sum_of_pathmags, " target=",target_sum_of_pathmags, " achieved=",achieved_sum_of_pathmags)
    #print("nPaths = ",npaths)
    #print("Num high-magnitude (|coeff|>%g, taylor<=%d) terms: %s" % (min_term_mag, calc.max_order, str([len(factors) for factors in factor_lists])))
    #print("Num FOAT: ",[len(inds) for inds in foat_indices_per_op])
    #print("---------------------------")

    #max_degrees = []
    #for i,factors in enumerate(factor_lists):
    #    max_degrees.append(max([f.coeff.degree() for f in factors]))
    #print("Max degrees = ",max_degrees)
    #for Ei,prp in enumerate(prps):
    #    print(Ei,":", prp.debug_report())
    #if db_paramvec is not None:
    #    for Ei,prp in enumerate(prps):
    #        print(Ei," => ", prp.evaluate(db_paramvec))

    #TODO: REMOVE - most of this is solved, but keep it around for another few commits in case we want to refer back to it.
    # - need to fill in some more details, namely how/where we hold weights and log-weights: in reps? in Term objs?  maybe consider Cython version?
    # need to consider how to perform "fastmode" in this... maybe need to traverse tree in some standard order?
    # what about having multiple thresholds for the different elabels... it seems good to try to run these calcs in parallel.
    # Note: may only need recusive tree traversal to consider incrementing positions *greater* than or equal to the one that was just incremented?
    #  (this may enforce some iteration ordering amenable to a fastmode calc)
    # Note2: when all effects have *same* op-part of terms, just different effect vector, then maybe we could split the effect into an op + effect
    #  to better use fastmode calc?  Or maybe if ordering is right this isn't necessary?
    #Add repcache as in cython version -- saves having to *construct* rep objects all the time... just update coefficients when needed instead?

    #... and we're done!
    return prps, sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)


# foat = first-order always-traversed
def traverse_paths_upto_threshold(oprep_lists, pathmag_threshold, num_elabels, foat_indices_per_op, fn_visitpath, debug=False):
    """ TODO: docstring """  # zot = zero-order-terms
    n = len(oprep_lists)
    nops = [len(oprep_list) for oprep_list in oprep_lists]
    b = [0] * n  # root
    log_thres = _np.log10(pathmag_threshold)

    ##TODO REMOVE
    #if debug:
    #    if debug > 1: print("BEGIN TRAVERSAL")
    #    accepted_bs_and_mags = {}

    def traverse_tree(root, incd, log_thres, current_mag, current_logmag, order):
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
                if oprep_lists[i][b[i] - 1].magnitude == 0:
                    mag = 0
                else:
                    mag = current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][b[i] - 1].magnitude)

                #TODO REMOVE
                #if debug:
                #    dbmags = [oprep_lists[k][b[k]].magnitude for k in range(n)]
                #    if debug>1: print("Accepting path: ",b, "(order",sub_order,"):", mag, '*'.join(map(str,dbmags))) #, logmag, " vs ", log_thres)
                #    assert(abs(_np.product(dbmags)-mag) < 1e-6)
                #    assert(tuple(b) not in accepted_bs_and_mags)
                #    accepted_bs_and_mags[tuple(b)] = mag
                fn_visitpath(b, mag, i)
                traverse_tree(b, i, log_thres, mag, logmag, sub_order)  # add any allowed paths beneath this one
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
                        mag = 0 if oprep_lists[i][orig_bi - 1].magnitude == 0 else \
                            current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][orig_bi - 1].magnitude)

                        fn_visitpath(b, mag, i)

                        #TODO REMOVE
                        #if debug:
                        #    sub_order = 1
                        #    dbmags = [oprep_lists[k][b[k]].magnitude for k in range(n)]
                        #    if debug>1: print("FOAT Accepting path: ",b, "(order",sub_order,"):", mag, '*'.join(map(str,dbmags))) #, logmag, " vs ", log_thres)
                        #    assert(abs(_np.product(dbmags)-mag) < 1e-6)
                        #    assert(tuple(b) not in accepted_bs_and_mags)
                        #    accepted_bs_and_mags[tuple(b)] = mag

                        if i != n - 1:
                            # if we're not incrementing (from a zero-order term) the final index, then we
                            # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                            orig_bn = b[n - 1]
                            for k in range(1, num_elabels):
                                b[n - 1] = k
                                mag2 = mag * (oprep_lists[n - 1][b[n - 1]].magnitude /
                                              oprep_lists[i][orig_bn].magnitude)
                                fn_visitpath(b, mag2, n - 1)

                                #TODO REMOVE
                                #if debug:
                                #    sub_order = 1
                                #    dbmags = [oprep_lists[k][b[k]].magnitude for k in range(n)]
                                #    if debug>1: print("FOAT Accepting path: ",b, "(order",sub_order,"):", mag2, '*'.join(map(str,dbmags))) #, logmag, " vs ", log_thres)
                                #    assert(abs(_np.product(dbmags)-mag2) < 1e-6)
                                #    assert(tuple(b) not in accepted_bs_and_mags)
                                #    accepted_bs_and_mags[tuple(b)] = mag2

                            b[n - 1] = orig_bn

                b[i] = orig_bi

                #TODO REMOVE
                #if debug and debug>1:
                #dbmags = [oprep_lists[k][b[k]].magnitude for k in range(n)]
                #mag = current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][b[i]-1].magnitude)
                #print("Rejected path: ",b, "(order",sub_order,"):", mag, '*'.join(map(str,dbmags))) #, logmag, " vs ", log_thres)
                #print(" --> logmag = ",logmag," thres=",log_thres)

            b[i] -= 1  # so we don't have to copy b
        #print("END: ",root)

    current_mag = 1.0; current_logmag = 0.0
    fn_visitpath(b, current_mag, 0)  # visit root (all 0s) path
    #if debug: accepted_bs_and_mags[tuple(b)] = current_mag TODO REMOVE
    traverse_tree(b, 0, log_thres, current_mag, current_logmag, 0)

    ##Step1: traverse first-order-always-traversed paths
    #for i in range(n):
    #    bi0 = b[i]
    #    base_wt = current_wt / oprep_lists[i][b[i]].weight
    #    while b[i]+1 < nAlwaysOn[i]:
    #        b[i] += 1
    #        wt = base_wt * oprep_lists[i][b[i]].weight
    #        fn_visitpath(b,wt)
    #        #print("Adding always-on path: ",b,' = ','*'.join(['%f' % term_weights[j][b[j]] for j in range(n)]),' = ',wt)
    #    b[i] = bi0

    #Step2: add additional paths
    #for i in reversed(range(n)):
    #    if b[i]+1 == nops[i]: continue
    #    b[i] += 1
    #    logmag = current_logmag + (oprep_lists[i][b[i]].logmagnitude - oprep_lists[i][b[i]-1].logmagnitude)
    #
    #    if logmag >= log_thres or b[i] <= num_foat_per_op[i]:
    #        if oprep_lists[i][b[i]-1].magnitude == 0:
    #            mag = 0
    #        else:
    #            mag = current_mag * (oprep_lists[i][b[i]].magnitude / oprep_lists[i][b[i]-1].magnitude)
    #        fn_visitpath(b, mag, i)
    #        traverse_tree(b, i, log_thres, mag, logmag, True) #add any allowed paths beneath this one
    #    b[i] -= 1 # so we don't have to copy b or root
    #print("END: ",root)
    return
    #TODO REMOVE
    #return accepted_bs_and_mags if debug else None


def pathmagnitude_threshold(oprep_lists, E_indices, nEffects, target_sum_of_pathmags, num_elabels=None, foat_indices_per_op=None,
                            initial_threshold=0.1, min_threshold=1e-10):
    """
    TODO: docstring - note: target_sum_of_pathmags is a *vector* that holds a separate value for each E-index
    """
    nIters = 0
    threshold = initial_threshold if (initial_threshold >= 0) else 0.1  # default value
    target_mag = target_sum_of_pathmags
    #print("Target magnitude: ",target_mag)
    threshold_upper_bound = 1.0
    threshold_lower_bound = None
    npaths_upper_bound = npaths_lower_bound = None
    #db_last_threshold = None #DEBUG TODO REMOVE
    #mag = 0; nPaths = 0

    if foat_indices_per_op is None:
        foat_indices_per_op = [()] * len(oprep_lists)

    def count_path(b, mg, incd):
        mag[E_indices[b[-1]]] += mg; nPaths[E_indices[b[-1]]] += 1

    while nIters < 100:  # TODO: allow setting max_nIters as an arg?
        mag = _np.zeros(nEffects, 'd')
        nPaths = _np.zeros(nEffects, int)
        accepted_bs_and_mags = traverse_paths_upto_threshold(
            oprep_lists, threshold, num_elabels, foat_indices_per_op, count_path)  # sets mag and nPaths

        ##TODO REMOVE
        #if db_last_threshold is not None:
        #    if db_last_mag + db_last_threshold * (nPaths[0] - db_last_nPaths) < mag[0]:
        #        print("Problem!")
        #        print(db_last_mag, db_last_threshold, db_last_nPaths, mag[0], threshold, nPaths[0])
        #        new_bs_and_mags = {}
        #        for x in accepted_bs_and_mags:
        #            if x not in db_last_accepted:
        #                new_bs_and_mags[x] = accepted_bs_and_mags[x]
        #        missing = set()
        #        for x in db_last_accepted:
        #            if x not in accepted_bs_and_mags:
        #                missing.add(x)
        #        print("New bs and mags (%d):" % len(new_bs_and_mags))
        #        print(new_bs_and_mags)
        #        print("Missing (%d):" % len(missing))
        #        print(missing)
        #        print("Sum current: ", sum(accepted_bs_and_mags.values()))
        #        print("Sum last: ", sum(db_last_accepted.values()))
        #
        #        mag = _np.zeros(nEffects,'d')
        #        nPaths = _np.zeros(nEffects,int) # db_last_threshold
        #        accepted_bs_and_mags = traverse_paths_upto_threshold(oprep_lists, threshold, num_elabels, foat_indices_per_op, count_path, debug=2)
        #        print(mag,nPaths)
        #        assert(False), "PROBLEM!"
        #db_last_mag = mag[0]
        #db_last_nPaths = nPaths[0]
        #db_last_threshold = threshold
        #db_last_accepted = accepted_bs_and_mags

        #TODO REMOVE
        #if _np.any(mag > target_mag + 0.0001):
        #    print("MAG SEEMS TOO HIGH!! - printing debug:",mag)
        #    max_check = 1.0
        #    for i,oprep_list in enumerate(oprep_lists[0:-1]):
        #        tmags = [t.magnitude for t in oprep_list[num_zoat_per_op[i]:num_zoat_per_op[i]+num_foat_per_op[i]]]
        #        sum_mags = sum(tmags)
        #        print("mags(%d):" % len(tmags), tmags, " -> sum_mags = ",sum_mags," -> expd -> ", _np.exp(sum_mags))
        #        max_check *= _np.exp(sum_mags)
        #    oprep_list = [ op for i,op in enumerate(oprep_lists[-1]) if E_indices[i] == 0 ]
        #    tmags = [t.magnitude for t in oprep_list[num_zoat_per_op[i]:num_zoat_per_op[i]+num_foat_per_op[i]]]
        #    sum_mags = sum(tmags)
        #    print("last mags(%d):" % len(tmags), tmags, " -> sum_mags = ",sum_mags," -> expd -> ", _np.exp(sum_mags))
        #    max_check *= _np.exp(sum_mags)
        #    print("max check = ",max_check)
        #
        #    max_check2 = 1.0
        #    for op in db_ops:
        #        tmags = [t.magnitude for t in op.get_taylor_order_terms(1)]
        #        print("Op first order mags (%d) = " % len(tmags), tmags, " -> sum_mags = ",sum(tmags)," -> expd -> ", _np.exp(sum(tmags)))
        #        print("Op mag = ",op.get_total_term_magnitude())
        #        max_check2 *= op.get_total_term_magnitude()
        #    print("max check 2 = ",max_check2)
        #
        #    for op in db_ops:
        #        print("OP")
        #        print("First order polys:")
        #        for t in op.get_taylor_order_terms(1):
        #            print(t.coeff)
        #        print("Second order polys:")
        #        for t in op.get_taylor_order_terms(2):
        #            print(t.coeff)
        #
        #    mag = _np.zeros(nEffects,'d')
        #    nPaths = _np.zeros(nEffects,int)
        #    traverse_paths_upto_threshold(oprep_lists, threshold, num_zoat_per_op, num_foat_per_op, count_path, debug=True) # sets mag and nPaths
        #    assert(False),"STOP"

        if _np.all(mag >= target_mag):  # try larger threshold
            threshold_lower_bound = threshold
            npaths_lower_bound = nPaths
            if threshold_upper_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold *= 2
        else:  # try smaller threshold
            threshold_upper_bound = threshold
            npaths_upper_bound = nPaths
            if threshold_lower_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold /= 2

        #print("  Interval: threshold in [%s,%s]: %s %s" % (str(threshold_upper_bound),str(threshold_lower_bound),mag,nPaths))
        if threshold_upper_bound is not None and threshold_lower_bound is not None and \
           (threshold_upper_bound - threshold_lower_bound) / threshold_upper_bound < 1e-3:
            #print("Converged after %d iters!" % nIters)
            break
        if threshold_upper_bound < min_threshold:  # could also just set min_threshold to be the lower bound initially?
            threshold_upper_bound = threshold_lower_bound = min_threshold
            break

        nIters += 1

    #Run path traversal once more to count final number of paths
    mag = _np.zeros(nEffects, 'd')
    nPaths = _np.zeros(nEffects, int)
    traverse_paths_upto_threshold(oprep_lists, threshold_lower_bound, num_elabels,
                                  foat_indices_per_op, count_path)  # sets mag and nPaths

    return threshold_lower_bound, nPaths, mag


def _unitary_sim_pre(complete_factors, comm, memLimit):
    rhoVec = complete_factors[0].pre_state  # a prep representation
    for f in complete_factors[0].pre_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.pre_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].post_ops:
        rhoVec = f.acton(rhoVec)

    EVec = complete_factors[-1].post_effect
    return EVec.amplitude(rhoVec)


def _unitary_sim_post(complete_factors, comm, memLimit):
    rhoVec = complete_factors[0].post_state  # a prep representation
    for f in complete_factors[0].post_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.post_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].pre_ops:
        rhoVec = f.acton(rhoVec)
    EVec = complete_factors[-1].pre_effect
    return _np.conjugate(EVec.amplitude(rhoVec))  # conjugate for same reason as above
