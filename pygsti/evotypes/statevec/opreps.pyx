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
import numpy as _np
import copy as _copy

import itertools as _itertools
from ...baseobjs.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs
from ...tools import basistools as _bt
from ...tools import optools as _ot

from scipy.sparse.linalg import LinearOperator


cdef class OpRep(_basereps_cython.OpRep):
    def __cinit__(self):
        self.c_rep = NULL
        self.state_space = None

    def __reduce__(self):
        return (OpRep, ())

    def __pygsti_reduce__(self):
        return self.__reduce__()

    def __dealloc__(self):
        if self.c_rep != NULL:
            del self.c_rep

    @property
    def dim(self):
        if self.c_rep == NULL:  # OpRepExpErrorgen case
            return self.state_space.udim
        else:
            return self.c_rep._dim

    def acton(self, StateRep state not None):
        cdef StateRep out_state = StateRepDensePure(_np.empty(self.c_rep._dim, dtype=_np.complex128),
                                                    state.basis, self.state_space)
        self.c_rep.acton(state.c_state, out_state.c_state)
        return out_state

    def adjoint_acton(self, StateRep state not None):
        cdef StateRep out_state = StateRepDensePure(_np.empty(self.c_rep._dim, dtype=_np.complex128),
                                                    state.basis, self.state_space)
        self.c_rep.adjoint_acton(state.c_state, out_state.c_state)
        return out_state

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRepDensePure(_np.ascontiguousarray(v, _np.complex128), None, self.state_space)
            return self.acton(in_state).to_dense('Hilbert')
        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:,0]
            in_state = StateRepDensePure(_np.ascontiguousarray(v, _np.complex128), None, self.state_space)
            return self.adjoint_acton(in_state).to_dense('Hilbert')
        dim = self.c_rep._dim
        return LinearOperator((dim,dim), matvec=mv, rmatvec=rmv, dtype=_np.complex128)


cdef class OpRepDenseUnitary(OpRep):
    cdef public _np.ndarray base
    cdef public object basis

    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.udim, _np.complex128)
        assert(mx.ndim == 2 and mx.shape[0] == state_space.udim)

        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        assert(self.c_rep == NULL)
        self.c_rep = new OpCRep_DenseUnitary(<double complex*>self.base.data,
                                             <INT>self.base.shape[0])
        self.state_space = state_space
        self.basis = basis

    def base_has_changed(self):
        pass

    def to_dense(self, on_space):
        if on_space in ('minimal', 'Hilbert'):
            return self.base
        elif on_space == 'HilbertSchmidt':
            return _bt.change_basis(_ot.unitary_to_process_mx(self.base), 'std', self.basis)
        else:
            raise ValueError("Invalid `on_space` argument: %s" % str(on_space))

    def __reduce__(self):
        # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
        # (so self.base *owns* it's data) and manually convey the writeable flag.
        return (OpRepDenseUnitary.__new__, (self.__class__,),
                (self.base, self.basis, self.state_space, self.base.flags.writeable))

    def __setstate__(self, state):
        data, basis, state_space, writable = state
        self.base = _np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.base.flags.writeable = writable
        self.basis = basis
        self.state_space = state_space

        assert(self.c_rep == NULL)  # if setstate is call, __init__ shouldn't have been
        self.c_rep = new OpCRep_DenseUnitary(<double complex*>self.base.data,
                                             <INT>self.base.shape[0])

    def __str__(self):
        s = ""
        cdef OpCRep_DenseUnitary* my_cgate = <OpCRep_DenseUnitary*>self.c_rep
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s

    def copy(self):
        return OpRepDenseUnitary(self.base.copy(), self.basis, self.state_space)


cdef class OpRepStandard(OpRepDenseUnitary):
    cdef public object name

    def __init__(self, name, basis, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        state_space = _StateSpace.cast(state_space)
        assert(U.shape[0] == state_space.udim)
        super(OpRepStandard, self).__init__(U, basis, state_space)

    def __reduce__(self):
        return (OpRepStandard, (self.name, self.basis, self.state_space))

    def __setstate__(self, state):
        pass  # must define this becuase base class does - need to override it


#class OpRepStochastic(OpRepDense):
# - maybe we could add this, but it wouldn't be a "dense" op here,
#   perhaps we need to change API?


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
        self.c_rep = new OpCRep_Composed(gate_creps, self.state_space.udim)

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
    cdef public object factor_reps  # list of OpRep objs?

    def __cinit__(self, factor_reps, state_space):
        self.factor_reps = factor_reps
        cdef INT i
        cdef INT nfactors = len(factor_reps)
        cdef vector[OpCRep*] factor_creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            factor_creps[i] = (<OpRep?>factor_reps[i]).c_rep
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Sum(factor_creps, self.state_space.udim)

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
            _np.array([state_space.label_udimension(l) for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] action_inds = _np.array(labelIndices, _np.int64)
        assert(_np.product([num_basis_els[i] for i in action_inds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        cdef INT dim = state_space.udim
        cdef INT nblocks = state_space.num_tensor_product_blocks
        cdef INT active_block_index = iTensorProdBlk
        cdef INT ncomponents_in_active_block = len(state_space.tensor_product_block_labels(active_block_index))
        cdef INT embedded_dim = embedded_rep.dim
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] blocksizes = \
            _np.array([_np.product(state_space.tensor_product_block_udimensions(k))
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
                                        active_block_index, nblocks, state_space.udim)
        self.state_space = state_space

    def copy(self):
        return _copy.deepcopy(self)  # I think this should work using reduce/setstate framework TODO - test and maybe put in base class?


cdef class OpRepRepeated(OpRep):
    cdef public OpRep repeated_rep
    cdef public INT num_repetitions

    def __cinit__(self, OpRep rep_to_repeat, INT num_repetitions, state_space):
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Repeated(self.repeated_rep.c_rep, num_repetitions, self.state_space.udim)

    def __reduce__(self):
        return (OpRepRepeated, (self.repeated_rep, self.num_repetitions, self.state_space))

    def copy(self):
        return OpRepRepeated(self.repeated_rep.copy(), self.num_repetitions, self.state_space.copy())


cdef class OpRepExpErrorgen(OpRep):
    cdef public object errorgen_rep

    def __init__(self, errorgen_rep):
        self.errorgen_rep = errorgen_rep
        self.state_space = errorgen_rep.state_space
        self.c_rep = NULL  # cannot act with this rep type for now
        # (really we could flesh it out more and fail acton() when the errorgen
        #  cannot be exponentiated/made dense?)

    def errgenrep_has_changed(self, onenorm_upperbound):
        pass

    def acton(self, StateRep state not None):
        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")

    def adjoint_acton(self, StateRep state not None):
        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")

    def __reduce__(self):
        return (OpRepExpErrorgen, (self.errorgen_rep,))

    def copy(self):
        return _copy.deepcopy(self)  # I think this should work using reduce/setstate framework TODO - test and maybe put in base class?


cdef class OpRepLindbladErrorgen(OpRep):
    cdef public object Lterms
    cdef public object Lterm_coeffs
    cdef public object LtermdictAndBasis

    def __cinit__(self, lindblad_term_dict, basis, state_space):
        self.Lterms = None
        self.Lterm_coeffs = None
        self.LtermdictAndBasis = (lindblad_term_dict, basis)
        self.state_space = state_space

    def acton(self, StateRep state not None):
        raise AttributeError("Cannot currently act with statevec.OpRepLindbladErrorgen - for terms only!")

    def adjoint_acton(self, StateRep state not None):
        raise AttributeError("Cannot currently act with statevec.OpRepLindbladErrorgen - for terms only!")

    def __reduce__(self):
        return (OpRepLindbladErrorgen, (self.LtermdictAndBasis[0], self.LtermdictAndBasis[1], self.state_space))
