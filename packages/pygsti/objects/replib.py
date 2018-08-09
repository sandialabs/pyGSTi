"""Defines Python-version calculation "representation" objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import sys
import numpy as _np
import scipy.sparse as _sps
import itertools as _itertools

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import matrixtools as _mt
from ..tools import listtools as _lt

# DEBUG!!!
DEBUG_FCOUNT = 0

class DMStateRep(object):
    def __init__(self, data):
        self.data = _np.asarray(data,'d')

    def copy_from(self, other):
        self.data = other.data.copy()

    def __str__(self):
        return str(self.data)


class DMEffectRep(object):
    def __init__(self,dim):
        self.dim = dim
    def probability(self, state):
        raise NotImplementedError()


class DMEffectRep_Dense(DMEffectRep):
    def __init__(self, data):
        self.data = _np.array(data,'d')
        super(DMEffectRep_Dense,self).__init__(len(self.data))
    def probability(self, state):
        # can assume state is a DMStateRep
        return _np.dot(self.data, state.data) # not vdot b/c *real* data

class DMEffectRep_TensorProd(DMEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim #Unused
        super(DMEffectRep_TensorProd,self).__init__(dim)

    def todense(self, outvec):
        N = self.dim
        #Put last factor at end of outvec
        k = self.nfactors-1  #last factor
        off = N-self.factor_dims[k] #offset into outvec
        for i in range(self.factor_dims[k]):
            outvec[off+i] = self.kron_array[k,i]
        sz = self.factor_dims[k]

        #Repeatedly scale&copy last "sz" elements of outputvec forward
        # (as many times as there are elements in the current factor array)
        # - but multiply *in-place* the last "sz" elements.
        for k in range(self.nfactors-2,-1,-1): #for all but the last factor
            off = N-sz*self.factor_dims[k]
            endoff = N-sz

            #For all but the final element of self.kron_array[k,:],
            # mult&copy final sz elements of outvec into position
            for j in range(self.factor_dims[k]-1):
                mult = self.kron_array[k,j]
                for i in range(sz):
                    outvec[off+i] = mult*outvec[endoff+i]
                off += sz

            #Last element: in-place mult
            #assert(off == endoff)
            mult = self.kron_array[k, self.factor_dims[k]-1]
            for i in range(sz):
                outvec[endoff+i] *= mult
            sz *= self.factor_dims[k]

        return outvec

    def probability(self, state): # allow scratch to be passed in?
        scratch = _np.empty(self.dim, 'd')
        Edense = self.todense( scratch  )
        return _np.dot(Edense,state.data) # not vdot b/c data is *real*


class DMEffectRep_Computational(DMEffectRep):
    def __init__(self, zvals, dim):
        # int dim = 4**len(zvals) -- just send as argument for speed?
        assert(dim == 4**len(zvals))
        assert(len(zvals) <= 64), "Cannot create a Computational basis rep with >64 qubits!"
          # Current storage of computational basis states converts zvals -> 64-bit integer

        base = 1
        self.zvals_int = 0
        for v in zvals:
            assert(v in (0,1)), "zvals must contain only 0s and 1s"
            self.zvals_int += base*v
            base *= 2 # or left shift?

        self.nfactors = len(zvals) # (or nQubits)
        self.abs_elval = 1/(_np.sqrt(2)**self.nfactors)

        super(DMEffectRep_Computational,self).__init__(dim)

    def parity(self, x):
        """recursively divide the (64-bit) integer into two equal
           halves and take their XOR until only 1 bit is left """
        x = (x & 0x00000000FFFFFFFF)^(x >> 32)
        x = (x & 0x000000000000FFFF)^(x >> 16)
        x = (x & 0x00000000000000FF)^(x >> 8)
        x = (x & 0x000000000000000F)^(x >> 4)
        x = (x & 0x0000000000000003)^(x >> 2)
        x = (x & 0x0000000000000001)^(x >> 1)
        return x & 1 # return the last bit (0 or 1)

    def todense(self, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if not trust_outvec_sparsity:
            outvec[:] = 0 #reset everything to zero

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
            finalIndx = sum([ 3*(4**(N-1-k)) for k in range(N) if bool(finds & (1<<k)) ])

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
        Edense = self.todense( scratch  )
        return _np.dot(Edense,state.data) # not vdot b/c data is *real*



class DMGateRep(object):
    def __init__(self,dim):
        self.dim = dim
    def acton(self, state):
        raise NotImplementedError()
    def adjoint_acton(self, state):
        raise NotImplementedError()


class DMGateRep_Dense(DMGateRep):
    def __init__(self, data):
        self.data = data
        super(DMGateRep_Dense,self).__init__(self.data.shape[0])
    def acton(self, state):
        return DMStateRep( _np.dot(self.data, state.data) )
    def adjoint_acton(self, state):
        return DMStateRep( _np.dot(self.data.T, state.data) ) # no conjugate b/c *real* data
    def __str__(self):
        return "DMGateRep_Dense:\n" + str(self.data)


class DMGateRep_Embedded(DMGateRep):
    def __init__(self, embedded_gate, numBasisEls, actionInds,
                 blocksizes, embedded_dim, nComponentsInActiveBlock,
                 iActiveBlock, nBlocks, dim):

        self.embedded_gate = embedded_gate
        self.numBasisEls = numBasisEls
        self.actionInds = actionInds
        self.blocksizes = blocksizes

        numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds: numBasisEls_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [ list(range(n)) for n in numBasisEls_noop_blankaction ]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array( _np.flipud( _np.cumprod([1] + list(
                                      reversed(list(numBasisEls[:-1])))) ), _np.int64)
        self.basisInds_action = [ list(range(numBasisEls[i])) for i in actionInds ]

        self.embeddedDim = embedded_dim
        self.nComponents = nComponentsInActiveBlock
        self.iActiveBlock = iActiveBlock
        self.nBlocks = nBlocks
        super(DMGateRep_Embedded,self).__init__(dim)

    def _acton_other_blocks_trivially(self, output_state,state):
        offset = 0
        for iBlk,blockSize in enumerate(self.blocksizes):
            if iBlk != self.iActiveBlock:
                output_state.data[offset:offset+blockSize] = state.data[offset:offset+blockSize] #identity op
            offset += blockSize

    def acton(self, state):
        output_state = DMStateRep( _np.zeros(state.data.shape, 'd') )
        offset = 0 #if relToBlock else self.offset (relToBlock == False here)

        #print("DB REPLIB ACTON: ",self.basisInds_noop_blankaction)
        #print("DB REPLIB ACTON: ",self.basisInds_action)
        #print("DB REPLIB ACTON: ",self.multipliers)
        for b in _itertools.product(*self.basisInds_noop_blankaction): #zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for gate_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i,bInd in zip(self.actionInds,gate_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i]*bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep( state.data[inds] )
            embedded_outstate= self.embedded_gate.acton( embedded_instate )
            output_state.data[ inds ] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state,state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = DMStateRep( _np.zeros(state.data.shape, 'd') )
        offset = 0 #if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction): #zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for gate_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i,bInd in zip(self.actionInds,gate_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i]*bInd
                inds.append(offset + vec_index)
            embedded_instate = DMStateRep( state[inds] )
            emedded_outstate= self.embedded_gate.adjoint_acton( embedded_instate )
            output_state.data[ inds ] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state,state)
        return output_state



class DMGateRep_Composed(DMGateRep):
    def __init__(self, factor_gate_reps):
        assert(len(factor_gate_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorgates = factor_gate_reps
        super(DMGateRep_Composed,self).__init__(factor_gate_reps[0].dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorgates:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate matrix on an input state """
        for gate in reversed(self.factorgates):
            state = gate.adjoint_acton(state)
        return state


class DMGateRep_Lindblad(DMGateRep):
    def __init__(self, A_data, A_indices, A_indptr,
                 mu, eta, m_star, s, unitarypost_data,
                 unitarypost_indices, unitarypost_indptr):
        dim = len(A_indptr)-1;
        self.A = _sps.csr_matrix( (A_data,A_indices,A_indptr), shape=(dim,dim) )
        if len(unitarypost_data) > 0: # (nnz > 0)
            self.unitary_postfactor = _sps.csr_matrix(
                (unitarypost_data,unitarypost_indices,
                 unitarypost_indptr), shape=(dim,dim) )
        else:
            self.unitary_postfactor = None # no unitary postfactor

        self.mu = mu
        self.eta = eta
        self.m_star = m_star
        self.s = s
        super(DMGateRep_Lindblad,self).__init__(dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        if self.unitary_postfactor is not None:
            statedata = self.unitary_postfactor.dot(state.data)
        else:
            statedata = state.data

        tol = 1e-16; # 2^-53 (=Scipy default) -- TODO: make into an arg?
        statedata = _mt._custom_expm_multiply_simple_core(
            self.A, statedata, self.mu, self.m_star, self.s, tol, self.eta)
        return DMStateRep(statedata)

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for sparse Lindblad Gate Reps yet.")


# State vector (SV) propagation wrapper classes
class SVStateRep(object):
    def __init__(self, data):
        self.data = _np.asarray(data,complex)

    def copy_from(self, other):
        self.data = other.data.copy()

    def __str__(self):
        return str(self.data)


class SVEffectRep(object):
    def __init__(self,dim):
        self.dim = dim
    def probability(self, state):
        return abs(self.amplitude(state))**2
    def amplitude(self, state):
        raise NotImplementedError()


class SVEffectRep_Dense(SVEffectRep):
    def __init__(self, data):
        self.data = _np.array(data,complex)
        super(SVEffectRep_Dense,self).__init__(len(self.data))
    def amplitude(self, state):
        # can assume state is a SVStateRep
        return _np.vdot(self.data, state.data) # (or just 'dot')

class SVEffectRep_TensorProd(SVEffectRep):
    def __init__(self, kron_array, factor_dims, nfactors, max_factor_dim, dim):
        # int dim = _np.product(factor_dims) -- just send as argument for speed?
        assert(dim == _np.product(factor_dims))
        self.kron_array = kron_array
        self.factor_dims = factor_dims
        self.nfactors = nfactors
        self.max_factor_dim = max_factor_dim #Unused
        super(SVEffectRep_TensorProd,self).__init__(dim)

    def todense(self, outvec):
        N = self.dim
        #Put last factor at end of outvec
        k = self.nfactors-1  #last factor
        off = N-self.factor_dims[k] #offset into outvec
        for i in range(self.factor_dims[k]):
            outvec[off+i] = self.kron_array[k,i]
        sz = self.factor_dims[k]

        #Repeatedly scale&copy last "sz" elements of outputvec forward
        # (as many times as there are elements in the current factor array)
        # - but multiply *in-place* the last "sz" elements.
        for k in range(self.nfactors-2,-1,-1): #for all but the last factor
            off = N-sz*self.factor_dims[k]
            endoff = N-sz

            #For all but the final element of self.kron_array[k,:],
            # mult&copy final sz elements of outvec into position
            for j in range(self.factor_dims[k]-1):
                mult = self.kron_array[k,j]
                for i in range(sz):
                    outvec[off+i] = mult*outvec[endoff+i]
                off += sz

            #Last element: in-place mult
            #assert(off == endoff)
            mult = self.kron_array[k, self.factor_dims[k]-1]
            for i in range(sz):
                outvec[endoff+i] *= mult
            sz *= self.factor_dims[k]

        return outvec

    def amplitude(self, state): # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.todense( scratch  )
        return _np.vdot(Edense,state.data)


class SVEffectRep_Computational(DMEffectRep):
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

        base = 2**(len(zvals)-1)
        self.nonzero_index = 0
        for k,v in enumerate(zvals):
            assert(v in (0,1)), "zvals must contain only 0s and 1s"
            self.nonzero_index += base*v
            base /= 2 # or right shift?


    def todense(self, outvec, trust_outvec_sparsity=False):
        # when trust_outvec_sparsity is True, assume we only need to fill in the
        # non-zero elements of outvec (i.e. that outvec is already zero wherever
        # this vector is zero).
        if not trust_outvec_sparsity:
            outvec[:] = 0 #reset everything to zero
        outvec[self.nonzero_index] = 1.0

    def amplitude(self, state): # allow scratch to be passed in?
        scratch = _np.empty(self.dim, complex)
        Edense = self.todense( scratch  )
        return _np.vdot(Edense,state.data)


class SVGateRep(object):
    def __init__(self,dim):
        self.dim = dim
    def acton(self, state):
        raise NotImplementedError()
    def adjoint_acton(self, state):
        raise NotImplementedError()


class SVGateRep_Dense(SVGateRep):
    def __init__(self, data):
        self.data = data
        super(SVGateRep_Dense,self).__init__(self.data.shape[0])
    def acton(self, state):
        return SVStateRep( _np.dot(self.data, state.data) )
    def adjoint_acton(self, state):
        return SVStateRep( _np.dot(_np.conjugate(self.data.T), state.data) )
    def __str__(self):
        return "SVGateRep_Dense:\n" + str(self.data)


class SVGateRep_Embedded(SVGateRep):
    # exactly the same as DM case
    def __init__(self, embedded_gate, numBasisEls, actionInds,
                 blocksizes, embedded_dim, nComponentsInActiveBlock,
                 iActiveBlock, nBlocks, dim):

        self.embedded_gate = embedded_gate
        self.numBasisEls = numBasisEls
        self.actionInds = actionInds
        self.blocksizes = blocksizes

        numBasisEls_noop_blankaction = numBasisEls.copy()
        for i in actionInds: numBasisEls_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [ list(range(n)) for n in numBasisEls_noop_blankaction ]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array( _np.flipud( _np.cumprod([1] + list(
                                      reversed(list(numBasisEls[:-1])))) ), _np.int64)
        self.basisInds_action = [ list(range(numBasisEls[i])) for i in actionInds ]

        self.embeddedDim = embedded_dim
        self.nComponents = nComponentsInActiveBlock
        self.iActiveBlock = iActiveBlock
        self.nBlocks = nBlocks
        super(SVGateRep_Embedded,self).__init__(dim)

    def _acton_other_blocks_trivially(self, output_state,state):
        offset = 0
        for iBlk,blockSize in enumerate(self.blocksizes):
            if iBlk != self.iActiveBlock:
                output_state.data[offset:offset+blockSize] = state.data[offset:offset+blockSize] #identity op
            offset += blockSize

    def acton(self, state):
        output_state = SVStateRep( _np.zeros(state.data.shape, complex) )
        offset = 0 #if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction): #zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for gate_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i,bInd in zip(self.actionInds,gate_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i]*bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep( state.data[inds] )
            embedded_outstate= self.embedded_gate.acton( embedded_instate )
            output_state.data[ inds ] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state,state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = SVStateRep( _np.zeros(state.data.shape, complex) )
        offset = 0 #if relToBlock else self.offset (relToBlock == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction): #zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for gate_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i,bInd in zip(self.actionInds,gate_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i]*bInd
                inds.append(offset + vec_index)
            embedded_instate = SVStateRep( state.data[inds] )
            embedded_outstate= self.embedded_gate.adjoint_acton( embedded_instate )
            output_state.data[ inds ] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state,state)
        return output_state



class SVGateRep_Composed(SVGateRep):
    # exactly the same as DM case
    def __init__(self, factor_gate_reps):
        assert(len(factor_gate_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorsgates = factor_gate_reps
        super(SVGateRep_Composed,self).__init__(factor_gate_reps[0].dim)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorgates:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate matrix on an input state """
        for gate in reversed(self.factorgates):
            state = gate.adjoint_acton(state)
        return state



# Stabilizer state (SB) propagation wrapper classes
class SBStateRep(object):
    def __init__(self, smatrix, pvectors, amps):
        from .stabilizer import StabilizerFrame as _StabilizerFrame
        self.sframe = _StabilizerFrame(smatrix,pvectors,amps)
          # just rely on StabilizerFrame class to do all the heavy lifting...

    def copy(self):
        cpy = SBStateRep(_np.zeros((0,0),_np.int64),None,None) # makes a dummy cpy.sframe
        cpy.sframe = self.sframe.copy() # a legit copy *with* qubit filers copied too
        return cpy

    def __str__(self):
        return "SBStateRep:\n" + str(self.sframe)


class SBEffectRep(object):
    def __init__(self, zvals):
        self.zvals = zvals

    def probability(self, state):
        return state.sframe.measurement_probability(self.zvals, check=True) # use check for now?

    def amplitude(self, state):
        return state.sframe.extract_amplitude(self.zvals)



class SBGateRep(object):
    def __init__(self, n):
        self.n = n # number of qubits
    def acton(self, state):
        raise NotImplementedError()
    def adjoint_acton(self, state):
        raise NotImplementedError()


class SBGateRep_Embedded(SBGateRep):
    def __init__(self, embedded_gate, n, qubits):
        self.embedded_gate = embedded_gate
        self.qubit_indices = qubits
        super(SBGateRep_Embedded,self).__init__(n)

    def acton(self, state):
        state = state.copy() # needed?
        state.sframe.push_view(self.qubit_indices)
        outstate = self.embedded_gate.acton( state ) # works b/c sfame has "view filters"
        state.sframe.pop_view() # return input state to original view
        outstate.sframe.pop_view()
        return outstate

    def adjoint_acton(self, state):
        state = state.copy() # needed?
        state.sframe.push_view(self.qubit_indices)
        outstate = self.embedded_gate.adjoint_acton( state ) # works b/c sfame has "view filters"
        state.sframe.pop_view() # return input state to original view
        outstate.sframe.pop_view()
        return outstate


class SBGateRep_Composed(SBGateRep):
    # exactly the same as DM case except .dim -> .n
    def __init__(self, factor_gate_reps):
        assert(len(factor_gate_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factorgates = factor_gate_reps
        super(SBGateRep_Composed,self).__init__(factor_gate_reps[0].n)

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factorgates:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate matrix on an input state """
        for gate in reversed(self.factorgates):
            state = gate.adjoint_acton(state)
        return state


class SBGateRep_Clifford(SBGateRep):
    def __init__(self, smatrix, svector, smatrix_inv, svector_inv, unitary):
        self.smatrix = smatrix
        self.svector = svector
        self.smatrix_inv = smatrix_inv
        self.svector_inv = svector_inv
        self.unitary = unitary
        super(SBGateRep_Clifford,self).__init__(smatrix.shape[0] // 2)

    def acton(self, state):
        """ Act this gate map on an input state """
        state = state.copy() # (copies any qubit filters in .sframe too)
        state.sframe.clifford_update(self.smatrix, self.svector, self.unitary)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate matrix on an input state """
        # Note: cliffords are unitary, so adjoint == inverse
        state = state.copy() # (copies any qubit filters in .sframe too)
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

    def __init__(self, int_coeffs, max_order, max_num_vars):
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

        super(PolyRep,self).__init__()
        if int_coeffs is not None:
            self.update(int_coeffs)

    @property
    def coeffs(self): # so we can convert back to python Polys
        """ The coefficient dictionary (with encoded integer keys) """
        return dict(self) #for compatibility w/C case which can't derive from dict...

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
        coeffs = { self._int_to_vinds(i): v for k,v in self.items() }
        if max_num_vars is not None: self.max_num_vars = max_num_vars
        if max_order is not None: self.max_order = max_order
        int_coeffs = { self._vinds_to_int(k): v for k,v in coeffs.items() }
        self.clear()
        self.update(int_coeffs)

    def _vinds_to_int(self, vinds):
        """ Maps tuple of variable indices to encoded int """
        assert(len(vinds) <= self.max_order), "max_order is too low!"
        ret = 0; m = 1
        for i in vinds: # last tuple index is most significant
            assert(i < self.max_num_vars), "Variable index exceed maximum!"
            ret += (i+1)*m
            m *= self.max_num_vars+1
        return ret

    def _int_to_vinds(self, indx):
        """ Maps encoded int to tuple of variable indices """
        ret = []
        while indx != 0:
            nxt = indx // (self.max_num_vars+1)
            i = indx - nxt*(self.max_num_vars+1)
            ret.append(i-1)
            indx = nxt
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
        for i,coeff in self.items():
            ivar = self._int_to_vinds(i)
            cnt = float(ivar.count(wrtParam))
            if cnt > 0:
                l = list(ivar)
                del l[ivar.index(wrtParam)]
                dcoeffs[ tuple(l) ] = cnt * coeff
        int_dcoeffs = { self._vinds_to_int(k): v for k,v in dcoeffs.items() }
        return PolyRep(int_dcoeffs, self.max_order, self.max_num_vars)

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
        for i,coeff in self.items():
            ivar = self._int_to_vinds(i)
            ret += coeff * _np.product( [variable_values[i] for i in ivar] )
        return ret

    def compact(self):
        """
        Returns a compact representation of this polynomial as a 
        `(variable_tape, coefficient_tape)` 2-tuple of 1D nupy arrays.

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
        iscomplex = any([ abs(_np.imag(x)) > 1e-12 for x in self.values() ])
        nTerms = len(self)
        vinds = [ self._int_to_vinds(i) for i in self.keys() ]
        nVarIndices = sum(map(len,vinds))
        vtape = _np.empty(1 + nTerms + nVarIndices, _np.int64) # "variable" tape
        ctape = _np.empty(nTerms, complex if iscomplex else 'd') # "coefficient tape"

        i = 0
        vtape[i] = nTerms; i+=1
        for iTerm,k in enumerate(sorted(self.keys())):
            v = self._int_to_vinds(k)
            l = len(v)
            ctape[iTerm] = self[k] if iscomplex else _np.real(self[k])
            vtape[i] = l; i += 1
            vtape[i:i+l] = v; i += l
        assert(i == len(vtape)), "Logic Error!"
        return vtape, ctape

    def copy(self):
        """
        Make a copy of this polynomial representation.
        
        Returns
        -------
        PolyRep
        """
        cpy = PolyRep(None, self.max_order, self.max_num_vars)
        cpy.update(self) # constructor expects dict w/var-index keys, not ints like self has
        return cpy

    def map_indices(self, mapfn):
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
        new_items = { self._vinds_to_int(mapfn(self._int_to_vinds(k))): v
                      for k,v in self.items() }
        self.clear()
        self.update(new_items)

    def mult(self,x):
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
        newpoly = PolyRep(None, self.max_order, self.max_num_vars) # assume same *fixed* max_order, even during mult
        for k1,v1 in self.items():
            for k2,v2 in x.items():
                k = newpoly._vinds_to_int(sorted(self._int_to_vinds(k1)+x._int_to_vinds(k2)))
                if k in newpoly: newpoly[k] += v1*v2
                else: newpoly[k] = v1*v2
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
            varstr = ""; last_i = None; n=0
            for i in sorted(vinds):
                if i == last_i: n += 1
                elif last_i is not None:
                    varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
                last_i = i
            if last_i is not None:
                varstr += "x%d%s" % (last_i, ("^%d" % n) if n > 1 else "")
            #print("DB: vinds = ",vinds, " varstr = ",varstr)
            if abs(self[k]) > 1e-4:
                termstrs.append( "%s%s" % (fmt(self[k]), varstr) )
        if len(termstrs) > 0:
            return " + ".join(termstrs)
        else: return "0"


    def __repr__(self):
        return "PolyRep[ " + str(self) + " ]"


    def __add__(self,x):
        newpoly = self.copy()
        if isinstance(x, PolyRep):
            assert(self.max_order == x.max_order and self.max_num_vars == x.max_num_vars)
            for k,v in x.items():
                if k in newpoly: newpoly[k] += v
                else: newpoly[k] = v
        else: # assume a scalar that can be added to values
            for k in newpoly:
                newpoly[k] += x
        return newpoly

    def __mul__(self,x):
        if isinstance(x, PolyRep):
            return self.mult_poly(x)
        else: # assume a scalar that can multiply values
            return self.mult_scalar(x)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __pow__(self,n):
        ret = PolyRep({0: 1.0}, self.max_order, self.max_num_vars)
        cur = self
        for i in range(int(np.floor(np.log2(n)))+1):
            rem = n % 2 #gets least significant bit (i-th) of n
            if rem == 1: ret *= cur # add current power of x (2^i) if needed
            cur = cur*cur # current power *= 2
            n //= 2 # shift bits of n right
        return ret

    def __copy__(self):
        return self.copy()



class SVTermRep(object):
    # just a container for other reps (polys, states, effects, and gates)
    def __init__(self, coeff, pre_state, post_state,
                 pre_effect, post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops


class SBTermRep(object):
    # exactly the same as SVTermRep
    # just a container for other reps (polys, states, effects, and gates)
    def __init__(self, coeff, pre_state, post_state,
                 pre_effect, post_effect, pre_ops, post_ops):
        self.coeff = coeff
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_effect = pre_effect
        self.post_effect = post_effect
        self.pre_ops = pre_ops
        self.post_ops = post_ops


## END CLASSES -- BEGIN CALC METHODS


def propagate_staterep(staterep, gatereps):
    ret = staterep
    for gaterep in gatereps:
        ret = gaterep.acton(ret)
    return ret


def DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, scratch=None): # TODO remove scratch

    cacheSize = evalTree.cache_size()
    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    ret = _np.empty((len(evalTree),len(elabels)),'d')

    #Get rho & rhoCache
    rho = rhoVec.torep('prep')
    rho_cache = [None]*cacheSize # so we can store (s,p) tuples in cache

    #Get gatereps and ereps now so we don't make unnecessary .torep() calls
    gatereps = { gl:calc._getgate(gl).torep() for gl in evalTree.gateLabels }
    ereps = [ E.torep('effect') for E in EVecs ]

    #REMOVE?? - want some way to speed tensorprod effect actions...
    #if self.evotype in ("statevec", "densitymx"):
    #    Escratch = _np.empty(self.dim,typ) # memory for E.todense() if it wants it

    #comm is currently ignored
    #TODO: if evalTree is split, distribute among processors
    for i in evalTree.get_evaluation_order():
        iStart,remainder,iCache = evalTree[i]
        if iStart is None:  init_state = rho
        else:               init_state = rho_cache[iStart] #[:,None]

        #OLD final_state = self.propagate_state(init_state, remainder)
        final_state = propagate_staterep(init_state, [gatereps[gl] for gl in remainder])
        if iCache is not None: rho_cache[iCache] = final_state # [:,0] #store this state in the cache

        for j,erep in enumerate(ereps):
            ret[i,j] = erep.probability(final_state) #outcome probability

    #print("DEBUG TIME: pr_cache(dim=%d, cachesize=%d) in %gs" % (self.dim, cacheSize,_time.time()-tStart)) #DEBUG

    return ret


def DM_compute_dpr_cache(calc, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):

    eps = 1e-7 #hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(calc.Np) if (wrtSlice is None) else _slct.indices(wrtSlice)
    nDerivCols = len(param_indices) # *all*, not just locally computed ones

    rhoVec,EVecs = calc._rhoEs_from_labels(rholabel, elabels)
    pCache = _np.empty((len(evalTree),len(elabels)),'d')
    dpr_cache  = _np.zeros((len(evalTree), len(elabels), nDerivCols),'d')

    #Get (extension-type) representation objects
    rhorep = calc.preps[rholabel].torep('prep')
    ereps = [ calc.effects[el].torep('effect') for el in elabels]
    gate_lookup = { lbl:i for i,lbl in enumerate(evalTree.gateLabels) } # gate labels -> ints for faster lookup
    gatereps = { i:calc._getgate(lbl).torep() for lbl,i in gate_lookup.items() }
    cacheSize = evalTree.cache_size()

    # create rho_cache (or use scratch)
    if scratch is None:
        rho_cache = [None]*cacheSize # so we can store (s,p) tuples in cache
    else:
        assert(len(scratch) == cacheSize)
        rho_cache = scratch

    pCache = DM_compute_pr_cache(calc, rholabel, elabels, evalTree, comm, rho_cache) # here scratch is used...

    all_slices, my_slice, owners, subComm = \
            _mpit.distribute_slice(slice(0,len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start #beginning of where my_param_indices results
                        # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = { i: st+ii for ii,i in enumerate(my_param_indices) }

    orig_vec = calc.to_vector().copy()
    for i in range(calc.Np):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            calc.from_vector(vec)
            dpr_cache[:,:,iFinal] = ( DM_compute_pr_cache(
                calc, rholabel, elabels, evalTree, subComm, rho_cache) - pCache)/eps
    calc.from_vector(orig_vec)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, dpr_cache,[], axes=2, comm=comm)

    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (self.Np, self.dim, cacheSize, len(evalTree), evalTree.get_num_applies(), _time.time()-tStart)) #DEBUG

    return dpr_cache


def SV_prs_as_polys(calc, rholabel, elabels, gatestring, comm=None, memLimit=None, fastmode=True):
    return _prs_as_polys(calc, rholabel, elabels, gatestring, comm, memLimit, fastmode)

def SB_prs_as_polys(calc, rholabel, elabels, gatestring, comm=None, memLimit=None, fastmode=True):
    return _prs_as_polys(calc, rholabel, elabels, gatestring, comm, memLimit, fastmode)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_polys(calc, rholabel, elabels, gatestring, comm=None, memLimit=None, fastmode=True):
    """
    Computes polynomials of the probabilities for multiple spam-tuples of `gatestring`
    
    Parameters
    ----------
    calc : GateTermCalculator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability 
        polynomials.

    gatestring : GateString
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
    #print("PRS_AS_POLY gatestring = ",gatestring)
    #print("DB: prs_as_polys(",spamTuple,gatestring,calc.max_order,")")

    mpv = calc.Np # max_poly_vars
    mpo = calc.max_order*2 #max_poly_order

    # Construct dict of gate term reps
    distinct_gateLabels = sorted(set(gatestring))
    gate_term_reps = { glbl: [ [t.torep(mpo,mpv,"gate") for t in calc._getgate(glbl).get_order_terms(order)]
                                      for order in range(calc.max_order+1) ]
                       for glbl in distinct_gateLabels }

    #Similar with rho_terms and E_terms, but lists
    rho_term_reps = [ [t.torep(mpo,mpv,"prep") for t in calc.preps[rholabel].get_order_terms(order)]
                      for order in range(calc.max_order+1) ]

    E_term_reps = []
    E_indices = []
    for order in range(calc.max_order+1):
        cur_term_reps = [] # the term reps for *all* the effect vectors
        cur_indices = [] # the Evec-index corresponding to each term rep
        for i,elbl in enumerate(elabels):
            term_reps = [t.torep(mpo,mpv,"effect") for t in calc.effects[elbl].get_order_terms(order) ]
            cur_term_reps.extend( term_reps )
            cur_indices.extend( [i]*len(term_reps) )
        E_term_reps.append( cur_term_reps )
        E_indices.append( cur_indices )


    ##DEBUG!!!
    #print("DB NEW gate terms = ")
    #for glbl,order_terms in gate_term_reps.items():
    #    print("GATE ",glbl)
    #    for i,termlist in enumerate(order_terms):
    #        print("ORDER %d" % i)
    #        for term in termlist:
    #            print("Coeff: ",str(term.coeff))



    #HERE DEBUG!!!
    global DEBUG_FCOUNT
    db_part_cnt = 0
    db_factor_cnt = 0
    #print("DB: pr_as_poly for ",str(tuple(map(str,gatestring))), " max_order=",calc.max_order)

    prps = [None]*len(elabels)  # an array in "bulk" mode? or Polynomial in "symbolic" mode?
    for order in range(calc.max_order+1):
        #print("DB: pr_as_poly order=",order)
        db_npartitions = 0
        for p in _lt.partition_into(order, len(gatestring)+2): # +2 for SPAM bookends
            #factor_lists = [ calc._getgate(glbl).get_order_terms(pi) for glbl,pi in zip(gatestring,p) ]
            factor_lists = [ rho_term_reps[p[0]]] + \
                           [ gate_term_reps[glbl][pi] for glbl,pi in zip(gatestring,p[1:-1]) ] + \
                           [ E_term_reps[p[-1]] ]
            factor_list_lens = list(map(len,factor_lists))
            Einds = E_indices[p[-1]] # specifies which E-vec index each of E_term_reps[p[-1]] corresponds to

            if any([len(fl)==0 for fl in factor_lists]): continue

            #print("DB partition = ",p, "listlens = ",[len(fl) for fl in factor_lists])
            if fastmode: # filter factor_lists to matrix-compose all length-1 lists
                leftSaved = [None]*(len(factor_lists)-1)  # saved[i] is state after i-th
                rightSaved = [None]*(len(factor_lists)-1) # factor has been applied
                coeffSaved = [None]*(len(factor_lists)-1)
                last_index = len(factor_lists)-1

                for incd,fi in _lt.incd_product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i,factorInd in enumerate(fi)]

                    if incd == 0: # need to re-evaluate rho vector
                        rhoVecL = factors[0].pre_state # Note: `factor` is a rep & so are it's ops
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
                        rhoVecL = leftSaved[incd-1]
                        rhoVecR = rightSaved[incd-1]
                        coeff = coeffSaved[incd-1]

                    # propagate left and right states, saving as we go
                    for i in range(incd,last_index):
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
                    # Note - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
                    for f in reversed(factors[-1].post_ops):
                        rhoVecL = f.adjoint_acton(rhoVecL)
                    E = factors[-1].post_effect # effect representation
                    pLeft = E.amplitude(rhoVecL)

                    #Same for pre_ops and rhoVecR
                    for f in reversed(factors[-1].pre_ops):
                        rhoVecR = f.adjoint_acton(rhoVecR)
                    E = factors[-1].pre_effect
                    pRight = _np.conjugate(E.amplitude(rhoVecR))

                    #print("DB PYTHON: final block: pLeft=",pLeft," pRight=",pRight)
                    res = coeff.mult(factors[-1].coeff)
                    res.scale( (pLeft * pRight) )
                    #print("DB PYTHON: result = ",res)
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx] #final "factor" index == E-vector index
                    if prps[Ei] is None: prps[Ei]  = res
                    else:                prps[Ei] += res # could add_inplace?
                    #print("DB PYHON: prps[%d] = " % Ei, prps[Ei])

            else: # non-fast mode
                last_index = len(factor_lists)-1
                for fi in _itertools.product(*[range(l) for l in factor_list_lens]):
                    #if len(fi) == 0 ...  #never happens TODO REMOVE
                    factors = [factor_lists[i][factorInd] for i,factorInd in enumerate(fi)]
                    res    = _functools.reduce(lambda x,y: x.mult(y), [f.coeff for f in factors])
                    pLeft  = _unitary_sim_pre(factors, comm, memLimit)
                    pRight = _unitary_sim_post(factors, comm, memLimit)
                             # if not self.unitary_evolution else 1.0
                    res.scale( (pLeft * pRight) )
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx] #final "factor" index == E-vector index
                    #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
                    if prps[Ei] is None:  prps[Ei]  = res
                    else:                 prps[Ei] += res # add_inplace?
                    #print("DB running prps[",Ei,"] =",prps[Ei])

            #DEBUG!!!
            db_nfactors = [len(l) for l in factor_lists]
            db_totfactors = _np.product(db_nfactors)
            db_factor_cnt += db_totfactors
            DEBUG_FCOUNT += db_totfactors
            db_part_cnt += 1
            #print("DB: pr_as_poly   partition=",p,"(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

    #print("DONE -> FCOUNT=",DEBUG_FCOUNT)
    return prps # can be a list of polys



def _unitary_sim_pre(complete_factors, comm, memLimit):
    rhoVec = complete_factors[0].pre_state # a prep representation
    for f in complete_factors[0].pre_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.pre_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec) # LEXICOGRAPHICAL VS MATRIX ORDER

    # Note - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
    for f in reversed(complete_factors[-1].post_ops):
        rhoVec = f.adjoint_acton(rhoVec)
    EVec = complete_factors[-1].post_effect
    return EVec.amplitude(rhoVec)


def _unitary_sim_post(complete_factors, comm, memLimit):
    rhoVec = complete_factors[0].post_state # a prep representation
    for f in complete_factors[0].post_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.post_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec) # LEXICOGRAPHICAL VS MATRIX ORDER

    # Note - can't propagate effects, but can act w/adjoint of post_ops in reverse order...
    for f in reversed(complete_factors[-1].pre_ops):
        rhoVec = f.adjoint_acton(rhoVec)
    EVec = complete_factors[-1].pre_effect
    return _np.conjugate(EVec.amplitude(rhoVec)) # conjugate for same reason as above
