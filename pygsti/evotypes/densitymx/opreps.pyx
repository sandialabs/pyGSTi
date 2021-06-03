# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import sys
import numpy as _np
import copy as _copy

import itertools as _itertools
from ...models.statespace import StateSpace as _StateSpace
from ...tools import optools as _ot
from ...tools import matrixtools as _mt
from ...tools import basistools as _bt
from ...tools import internalgates as _itgs
from ...tools import lindbladtools as _lbt
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


cdef class OpRep(_basereps_cython.OpRep):
    def __cinit__(self):
        self.c_rep = NULL
        self.state_space = None

    def __reduce__(self):
        return (OpRep, ())

    def __dealloc__(self):
        if self.c_rep != NULL:
            del self.c_rep

    @property
    def dim(self):
        return self.state_space.dim

    def acton(self, StateRep state not None):
        cdef StateRep out_state = StateRepDense(_np.empty(self.c_rep._dim, dtype='d'), self.state_space)
        #print("PYX acton called w/dim ", self.c_rep._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_rep.acton(state.c_state, out_state.c_state)
        return out_state

    def adjoint_acton(self, StateRep state not None):
        cdef StateRep out_state = StateRepDense(_np.empty(self.c_rep._dim, dtype='d'), self.state_space)
        #print("PYX acton called w/dim ", self.c_rep._dim, out_state.c_state._dim)
        # assert(state.c_state._dataptr != out_state.c_state._dataptr) # DEBUG
        self.c_rep.adjoint_acton(state.c_state, out_state.c_state)
        return out_state

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRep(_np.ascontiguousarray(v,'d'))
            return self.acton(in_state).to_dense()
        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRep(_np.ascontiguousarray(v,'d'))
            return self.adjoint_acton(in_state).to_dense()
        dim = self.c_rep._dim
        return ScipyLinearOperator((dim,dim), matvec=mv, rmatvec=rmv, dtype='d') # transpose, adjoint, dot, matmat?


cdef class OpRepDense(OpRep):
    cdef public _np.ndarray base

    def __init__(self, mx, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.dim, 'd')
        assert(mx.ndim == 2 and mx.shape[0] == state_space.dim)

        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_Dense(<double*>self.base.data,
                                     <INT>self.base.shape[0])
        self.state_space = state_space

    def base_has_changed(self):
        pass

    def to_dense(self):
        return self.base

    def __reduce__(self):
        # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
        # (so self.base *owns* it's data) and manually convey the writeable flag.
        return (OpRepDense.__new__, (self.__class__,), (self.state_space, self.base, self.base.flags.writeable))

    def __setstate__(self, state):
        assert(self.c_rep == NULL)

        state_space, data, writable = state
        self.state_space = state_space
        self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.base.flags.writeable = writable

        assert(self.c_rep == NULL)  # if setstate is call, __init__ shouldn't have been
        self.c_rep = new OpCRep_Dense(<double*>self.base.data,
                                     <INT>self.base.shape[0])

    def __str__(self):
        s = ""
        cdef OpCRep_Dense* my_cgate = <OpCRep_Dense*>self.c_rep  # b/c we know it's a _Dense gate...
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s

    def copy(self):
        return OpRepDense(self.base.copy(), self.state_space)


cdef class OpRepSparse(OpRep):
    cdef public _np.ndarray data
    cdef public _np.ndarray indices
    cdef public _np.ndarray indptr

    def __cinit__(self, _np.ndarray[double, ndim=1, mode='c'] a_data,
                  _np.ndarray[_np.int64_t, ndim=1, mode='c'] a_indices,
                  _np.ndarray[_np.int64_t, ndim=1, mode='c'] a_indptr,
                  state_space):
        self.data = a_data
        self.indices = a_indices
        self.indptr = a_indptr
        cdef INT nnz = a_data.shape[0]
        cdef INT dim = a_indptr.shape[0]-1
        self.c_rep = new OpCRep_Sparse(<double*>a_data.data, <INT*>a_indices.data,
                                       <INT*>a_indptr.data, nnz, dim)

        state_space = _StateSpace.cast(state_space)
        assert(state_space.dim == dim)
        self.state_space = state_space

    def __reduce__(self):
        return (OpRepSparse, (self.data, self.indices, self.indptr, self.state_space))

    def copy(self):
        return OpRepSparse(self.data.copy(), self.indices.copy(), self.indptr.copy(), self.state_space.copy())


cdef class OpRepStandard(OpRepDense):
    cdef public object name
    cdef public object basis

    def __init__(self, name, basis, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)
        self.basis = basis

        U = std_unitaries[self.name]
        superop = _bt.change_basis(_ot.unitary_to_process_mx(U), 'std', basis)
        state_space = _StateSpace.cast(state_space)
        assert(superop.shape[0] == state_space.dim)

        super(OpRepStandard, self).__init__(superop, state_space)

    def __reduce__(self):
        return (OpRepStandard, (self.name, self.basis, self.state_space))


cdef class OpRepStochastic(OpRepDense):
    cdef public object basis
    cdef public object stochastic_superops

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state, state_space):
        self.basis = basis
        self.stochastic_superops = []
        for b in self.basis.elements[1:]:
            std_superop = _lbt.nonham_lindbladian(b, b, sparse=False)
            self.stochastic_superops.append(_bt.change_basis(std_superop, 'std', self.basis))

        state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == state_space.dim)

        super(OpRepStochastic, self).__init__(None, state_space)
        self.update_rates(initial_rates)

    def update_rates(self, rates):
        errormap = _np.identity(self.basis.dim)
        for rate, ss in zip(rates, self.stochastic_superops):
            errormap += rate * ss
        self.base[:, :] = errormap

    def to_dense(self):  # TODO - put this in all reps?  - used in stochastic op...
        # DEFAULT: raise NotImplementedError('No to_dense implemented for evotype "%s"' % self._evotype)
        return self.base  # copy?


cdef class OpRepComposed(OpRep):
    cdef public object factor_reps  # list of OpRep objs?

    def __cinit__(self, factor_op_reps, state_space):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[OpCRep*] gate_creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<OpRep?>factor_op_reps[i]).c_rep
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Composed(gate_creps, self.state_space.dim)

    def __reduce__(self):
        return (OpRepComposed, (self.factor_reps, self.state_space))

    def reinit_factor_op_reps(self, new_factor_op_reps):
        cdef INT i
        cdef INT nfactors = len(new_factor_op_reps)
        cdef vector[OpCRep*] creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            creps[i] = (<OpRep?>new_factor_op_reps[i]).c_rep
        (<OpCRep_Composed*>self.c_rep).reinit_factor_op_creps(creps)

    def copy(self):
        return OpRepComposed([f.copy() for f in self.factor_reps], self.c_rep._dim)


cdef class OpRepSum(OpRep):
    cdef public object factor_reps # list of OpRep objs?

    def __cinit__(self, factor_reps, state_space):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[OpCRep*] factor_creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<OpRep?>factor_reps[i]).c_rep
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Sum(factor_creps, self.state_space.dim)

    def __reduce__(self):
        return (OpRepSum, (self.factor_reps, self.state_space))

    def copy(self):
        return OpRepSum([f.copy() for f in self.factor_reps], self.c_rep._dim)


cdef class OpRepEmbedded(OpRep):
    cdef _np.ndarray noop_incrementers
    cdef _np.ndarray num_basis_els_noop_blankaction
    cdef _np.ndarray baseinds
    cdef _np.ndarray blocksizes
    cdef _np.ndarray num_basis_els
    cdef _np.ndarray action_inds
    cdef public OpRep embedded_rep

    def __init__(self, state_space, target_labels, OpRep embedded_rep):

        state_space = _StateSpace.cast(state_space)
        iTensorProdBlks = [state_space.label_tensor_product_block_index(label) for label in target_labels]
        # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit operation must correspond to the"
                             " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

        iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
        tensorProdBlkLabels = state_space.tensor_product_block_labels(iTensorProdBlk)
        # count possible *density-matrix-space* indices of each component of the tensor product block
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] num_basis_els = \
            _np.array([state_space.label_dimension(l) for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] action_inds = _np.array(labelIndices, _np.int64)
        assert(_np.product([num_basis_els[i] for i in action_inds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        cdef INT dim = state_space.dim
        cdef INT nblocks = state_space.num_tensor_product_blocks
        cdef INT active_block_index = iTensorProdBlk
        cdef INT ncomponents_in_active_block = len(state_space.tensor_product_block_labels(active_block_index))
        cdef INT embedded_dim = embedded_rep.dim
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] blocksizes = \
            _np.array([_np.product(state_space.tensor_product_block_dimensions(k))
                       for k in range(nblocks)], _np.int64)
        cdef INT i, j

        # num_basis_els_noop_blankaction is just num_basis_els with action_inds == 1
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] num_basis_els_noop_blankaction = num_basis_els.copy()
        for i in action_inds:
            num_basis_els_noop_blankaction[i] = 1 # for indexing the identity space

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        cdef _np.ndarray tmp = _np.empty(ncomponents_in_active_block,_np.int64)
        tmp[0] = 1
        for i in range(1,ncomponents_in_active_block):
            tmp[i] = num_basis_els[ncomponents_in_active_block-i]
        multipliers = _np.array( _np.flipud( _np.cumprod(tmp) ), _np.int64)

        # noop_incrementers[i] specifies how much the overall vector index
        #  is incremented when the i-th "component" digit is advanced
        cdef INT dec = 0
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] noop_incrementers = _np.empty(ncomponents_in_active_block,_np.int64)
        for i in range(ncomponents_in_active_block-1,-1,-1):
            noop_incrementers[i] = multipliers[i] - dec
            dec += (num_basis_els_noop_blankaction[i]-1)*multipliers[i]

        cdef INT vec_index
        cdef INT offset = 0 #number of basis elements preceding our block's elements
        for i in range(active_block_index):
            offset += blocksizes[i]

        # self.baseinds specifies the contribution from the "active
        #  component" digits to the overall vector index.
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] baseinds = _np.empty(embedded_dim,_np.int64)
        basisInds_action = [ list(range(num_basis_els[i])) for i in action_inds ]
        for ii,op_b in enumerate(_itertools.product(*basisInds_action)):
            vec_index = offset
            for j,bInd in zip(action_inds,op_b):
                vec_index += multipliers[j]*bInd
            baseinds[ii] = vec_index

        # Need to hold data references to any arrays used by C-type
        self.noop_incrementers = noop_incrementers
        self.num_basis_els_noop_blankaction = num_basis_els_noop_blankaction
        self.baseinds = baseinds
        self.blocksizes = blocksizes
        self.num_basis_els = num_basis_els
        self.action_inds = action_inds
        self.embedded_rep = embedded_rep # needed to prevent garbage collection?

        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_Embedded(embedded_rep.c_rep,
                                        <INT*>noop_incrementers.data, <INT*>num_basis_els_noop_blankaction.data,
                                        <INT*>baseinds.data, <INT*>blocksizes.data,
                                        embedded_dim, ncomponents_in_active_block,
                                        active_block_index, nblocks, dim)
        self.state_space = state_space

    def __reduce__(self):
        #TODO - fix this - need to call init properly and set self.state_space and (?) carry over target labels, etc?
        state = (self.noop_incrementers, self.num_basis_els_noop_blankaction, self.baseinds,
                 self.blocksizes, self.num_basis_els, self.action_inds, self.embedded_rep,
                 (<OpCRep_Embedded*>self.c_rep)._embeddedDim,
                 (<OpCRep_Embedded*>self.c_rep)._nComponents,
                 (<OpCRep_Embedded*>self.c_rep)._iActiveBlock,
                 (<OpCRep_Embedded*>self.c_rep)._nBlocks,
                 self.state_space)
        return (OpRepEmbedded.__new__, (self.__class__,), state)

    def __setstate__(self, state):
        (noop_incrementers, num_basis_els_noop_blankaction, baseinds,
         blocksizes, num_basis_els, action_inds, embedded_rep, embedded_dim,
         ncomponents_in_active_block, active_block_index, nblocks, state_space) = state

        self.noop_incrementers = noop_incrementers
        self.num_basis_els_noop_blankaction = num_basis_els_noop_blankaction
        self.baseinds = baseinds
        self.blocksizes = blocksizes
        self.num_basis_els = num_basis_els
        self.action_inds = action_inds
        self.embedded_rep = embedded_rep # needed to prevent garbage collection?

        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_Embedded(self.embedded_rep.c_rep,
                                        <INT*>self.noop_incrementers.data, <INT*>self.num_basis_els_noop_blankaction.data,
                                        <INT*>self.baseinds.data, <INT*>self.blocksizes.data,
                                        embedded_dim, ncomponents_in_active_block,
                                        active_block_index, nblocks, state_space.dim)
        self.state_space = state_space

    def copy(self):
        return _copy.deepcopy(self)  # I think this should work using reduce/setstate framework TODO - test and maybe put in base class?


cdef class OpRepExpErrorgen(OpRep):
    cdef public object errorgen_rep

    def __init__(self, errorgen_rep):
        self.errorgen_rep = errorgen_rep
        cdef INT dim = errorgen_rep.dim
        cdef double mu = 1.0
        cdef double eta = 1.0
        cdef INT m_star = 0
        cdef INT s = 0

        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_ExpErrorgen((<OpRep?>errorgen_rep).c_rep,
                                            mu, eta, m_star, s, dim)
        self.state_space = errorgen_rep.state_space

    def errgenrep_has_changed(self, onenorm_upperbound):
        # don't reset matrix exponential params (based on operator norm) when vector hasn't changed much
        mu, m_star, s, eta = _mt.expop_multiply_prep(
            self.errorgen_rep.aslinearoperator(),
            a_1_norm=onenorm_upperbound)
        self.set_exp_params(mu, eta, m_star, s)

    def set_exp_params(self, double mu, double eta, INT m_star, INT s):
        (<OpCRep_ExpErrorgen*>self.c_rep)._mu = mu
        (<OpCRep_ExpErrorgen*>self.c_rep)._eta = eta
        (<OpCRep_ExpErrorgen*>self.c_rep)._m_star = m_star
        (<OpCRep_ExpErrorgen*>self.c_rep)._s = s

    def get_exp_params(self):
        return ( (<OpCRep_ExpErrorgen*>self.c_rep)._mu,
                 (<OpCRep_ExpErrorgen*>self.c_rep)._eta,
                 (<OpCRep_ExpErrorgen*>self.c_rep)._m_star,
                 (<OpCRep_ExpErrorgen*>self.c_rep)._s)

    def __reduce__(self):
        return (OpRepExpErrorgen.__new__, (self.__class__,), (self.errorgen_rep,
                                                              (<OpCRep_ExpErrorgen*>self.c_rep)._mu,
                                                              (<OpCRep_ExpErrorgen*>self.c_rep)._eta,
                                                              (<OpCRep_ExpErrorgen*>self.c_rep)._m_star,
                                                              (<OpCRep_ExpErrorgen*>self.c_rep)._s))

    def __setstate__(self, state):
        errorgen_rep, mu, eta, m_star, s = state
        dim = errorgen_rep.dim
        self.errorgen_rep = errorgen_rep
        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_ExpErrorgen((<OpRep?>errorgen_rep).c_rep,
                                            mu, eta, m_star, s, dim)
        self.state_space = errorgen_rep.state_space
        
    def copy(self):
        return _copy.deepcopy(self)  # I think this should work using reduce/setstate framework TODO - test and maybe put in base class?


cdef class OpRepRepeated(OpRep):
    cdef public OpRep repeated_rep
    cdef public INT num_repetitions

    def __cinit__(self, OpRep rep_to_repeat, INT num_repetitions, state_space):
        state_space = _StateSpace.cast(state_space)
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        self.c_rep = new OpCRep_Repeated(self.repeated_rep.c_rep, num_repetitions, state_space.dim)
        self.state_space = state_space

    def __reduce__(self):
        return (OpRepRepeated, (self.repeated_rep, self.num_repetitions, self.state_space))

    def copy(self):
        return OpRepRepeated(self.repeated_rep.copy(), self.num_repetitions, self.state_space.copy())

