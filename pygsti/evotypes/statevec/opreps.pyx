# encoding: utf-8
# cython: profile=True
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
import time as pytime
import numpy as _np
import copy as _copy

import itertools as _itertools
from ...tools import mpitools as _mpit
from ...tools import slicetools as _slct
from ...tools import optools as _ot
from ...tools import optools as _mt
from ...tools import basistools as _bt
from ...tools import internalgates as _itgs
from ...tools import lindbladtools as _lbt
from scipy.sparse.linalg import LinearOperator

cdef double LARGE = 1000000000
# a large number such that LARGE is
# a very high term weight which won't help (at all) a
# path get included in the selected set of paths.

cdef double SMALL = 1e-5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


cdef class OpRep(_basereps_cython.OpRep):
    def __cinit__(self):
        self.c_rep = NULL

    def __reduce__(self):
        return (OpRep, ())

    def __dealloc__(self):
        if self.c_rep != NULL:
            del self.c_rep

    @property
    def dim(self):
        return self.c_rep._dim

    def acton(self, StateRep state not None):
        cdef StateRep out_state = StateRep(_np.empty(self.c_rep._dim, dtype=_np.complex128))
        self.c_rep.acton(state.c_state, out_state.c_state)
        return out_state

    def adjoint_acton(self, StateRep state not None):
        cdef StateRep out_state = StateRep(_np.empty(self.c_rep._dim, dtype=_np.complex128))
        self.c_rep.adjoint_acton(state.c_state, out_state.c_state)
        return out_state

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRep(_np.ascontiguousarray(v, _np.complex128))
            return self.acton(in_state).to_dense()
        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRep(_np.ascontiguousarray(v, _np.complex128))
            return self.adjoint_acton(in_state).to_dense()
        dim = self.c_rep._dim
        return LinearOperator((dim,dim), matvec=mv, rmatvec=rmv, dtype=_np.complex128)


cdef class OpRepPure(OpRep):
    cdef public _np.ndarray base

    def __init__(self, int dim):
        self.base = _np.require(_np.identity(dim, _np.complex128),
                                requirements=['OWNDATA', 'C_CONTIGUOUS'])
        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_Pure(<double complex*>self.base.data,
                                     <INT>self.base.shape[0])

    def __reduce__(self):
        # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
        # (so self.base *owns* it's data) and manually convey the writeable flag.
        return (OpRepPure.__new__, (self.__class__,), (self.base, self.base.flags.writeable))

    def __setstate__(self, state):
        assert(self.c_rep == NULL)

        data, writable = state
        self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.base.flags.writeable = writable

        assert(self.c_rep == NULL)  # if setstate is call, __init__ shouldn't have been
        self.c_rep = new OpCRep_Pure(<double complex*>self.base.data,
                                     <INT>self.base.shape[0])

    def __str__(self):
        s = ""
        cdef OpCRep_Pure* my_cgate = <OpCRep_Pure*>self.c_rep
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s

    def copy(self):
        cpy = OpRepPure(self.base.shape[0])
        cpy.base[:, :] = self.base.copy()
        return cpy


class OpRepStandard(OpRepPure):
    def __init__(self, name):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        super(OpRepStandard, self).__init__(U.shape[0])
        self.base[:, :] = U

    def __reduce__(self):
        return (OpRepStandard, (self.name,))


#class OpRepStochastic(OpRepDense):
# - maybe we could add this, but it wouldn't be a "dense" op here,
#   perhaps we need to change API?


cdef class OpRepComposed(OpRep):
    cdef public object factor_reps  # list of OpRep objs?

    def __cinit__(self, factor_op_reps, INT dim):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[OpCRep*] gate_creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            gate_creps[i] = (<OpRep?>factor_op_reps[i]).c_rep
        self.c_rep = new OpCRep_Composed(gate_creps, dim)

    def __reduce__(self):
        return (OpRepComposed, (self.factor_reps, self.c_rep._dim))

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

    def __cinit__(self, factor_reps, INT dim):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[OpCRep*] factor_creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<OpRep?>factor_reps[i]).c_rep
        self.c_rep = new OpCRep_Sum(factor_creps, dim)

    def __reduce__(self):
        return (OpRepSum, (self.factor_reps, self.c_rep._dim))

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

    def __init__(self, state_space_labels, target_labels, OpRep embedded_rep):

        iTensorProdBlks = [state_space_labels.tpb_index[label] for label in target_labels]
        # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit operation must correspond to the"
                             " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

        iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
        tensorProdBlkLabels = state_space_labels.labels[iTensorProdBlk]
        # count possible *density-matrix-space* indices of each component of the tensor product block
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] num_basis_els = \
            _np.array([state_space_labels.labeldims[l] for l in tensorProdBlkLabels], _np.int64)
        
        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] action_inds = _np.array(labelIndices, _np.int64)
        assert(_np.product([num_basis_els[i] for i in action_inds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        cdef INT dim = state_space_labels.dim
        cdef INT nblocks = state_space_labels.num_tensor_prod_blocks()
        cdef INT active_block_index = iTensorProdBlk
        cdef INT ncomponents_in_active_block = len(state_space_labels.labels[active_block_index])
        cdef INT embedded_dim = embedded_rep.dim
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] blocksizes = \
            _np.array([_np.product(state_space_labels.tensor_product_block_dims(k))
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


    def __reduce__(self):
        state = (self.noop_incrementers, self.num_basis_els_noop_blankaction, self.baseinds,
                 self.blocksizes, self.num_basis_els, self.action_inds, self.embedded_rep,
                 (<OpCRep_Embedded*>self.c_rep)._embeddedDim,
                 (<OpCRep_Embedded*>self.c_rep)._nComponents,
                 (<OpCRep_Embedded*>self.c_rep)._iActiveBlock,
                 (<OpCRep_Embedded*>self.c_rep)._nBlocks,
                 self.c_rep._dim)
        return (OpRepEmbedded, (), state)

    def __setstate__(self, state):
        (noop_incrementers, num_basis_els_noop_blankaction, baseinds,
         blocksizes, num_basis_els, action_inds, embedded_rep, embedded_dim,
         ncomponents_in_active_block, active_block_index, nblocks, dim) = state

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
                                        active_block_index, nblocks, dim)

    def copy(self):
        return _copy.deepcopy(self)  # I think this should work using reduce/setstate framework TODO - test and maybe put in base class?


cdef class OpRepRepeated(OpRep):
    cdef public OpRep repeated_rep
    cdef public INT num_repetitions

    def __cinit__(self, OpRep rep_to_repeat, INT num_repetitions, INT dim):
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        self.c_rep = new OpCRep_Repeated(self.repeated_rep.c_rep, num_repetitions, dim)

    def __reduce__(self):
        return (OpRepRepeated, (self.repeated_rep, self.num_repetitions, self.c_rep._dim))

    def copy(self):
        return OpRepRepeated(self.repeated_rep.copy(), self.num_repetitions, self.c_rep._dim)

