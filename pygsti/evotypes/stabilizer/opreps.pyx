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

from ...models.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs
from ...tools import symplectic as _symp


cdef class OpRep(_basereps_cython.OpRep):
    def __cinit__(self):
        self.c_rep = NULL
        self.state_space = None

    def __reduce__(self):
        return (OpRep, ())

    def __dealloc__(self):
        del self.c_rep

    @property
    def nqubits(self):
        return self.state_space.num_qubits

    #@property
    #def dim(self):
    #    return 2**(self.nqubits)  # assume "unitary evolution"-type mode

    def acton(self, StateRep state not None):
        cdef INT n = self.c_rep._n
        cdef INT namps = state.c_state._namps
        cdef StateRep out_state = StateRep(_np.empty((2*n,2*n), dtype=_np.int64),
                                               _np.empty((namps,2*n), dtype=_np.int64),
                                               _np.empty(namps, dtype=_np.complex128))
        self.c_rep.acton(state.c_state, out_state.c_state)
        return out_state

    def adjoint_acton(self, StateRep state not None):
        cdef INT n = self.c_rep._n
        cdef INT namps = state.c_state._namps
        cdef StateRep out_state = StateRep(_np.empty((2*n,2*n), dtype=_np.int64),
                                               _np.empty((namps,2*n), dtype=_np.int64),
                                               _np.empty(namps, dtype=_np.complex128))
        self.c_rep.adjoint_acton(state.c_state, out_state.c_state)
        return out_state


cdef class OpRepClifford(OpRep):
    cdef public _np.ndarray smatrix
    cdef public _np.ndarray svector
    cdef public _np.ndarray unitary
    cdef public _np.ndarray smatrix_inv
    cdef public _np.ndarray svector_inv
    cdef public _np.ndarray unitary_dagger

    def __cinit__(self, _np.ndarray[_np.complex128_t, ndim=2, mode='c'] unitarymx, symplecticrep, state_space):
        if symplecticrep is not None:
            self.smatrix, self.svector = symplecticrep
        else:
            # compute symplectic rep from unitary
            self.smatrix, self.svector = _symp.unitary_to_symplectic(self.unitary, flagnonclifford=True)

        self.smatrix_inv, self.svector_inv = _symp.inverse_clifford(
            self.smatrix, self.svector)  # cache inverse since it's expensive

        self.unitary = unitarymx
        self.unitary_dagger = _np.ascontiguousarray(_np.conjugate(_np.transpose(unitarymx)))

        #Make sure all arrays are contiguous
        self.smatrix = _np.ascontiguousarray(self.smatrix)
        self.svector = _np.ascontiguousarray(self.svector)
        self.smatrix_inv = _np.ascontiguousarray(self.smatrix_inv)
        self.svector_inv = _np.ascontiguousarray(self.svector_inv)

        self.state_space = _StateSpace.cast(state_space)
        assert(self.state_space.num_qubits == self.smatrix.shape[0] // 2)
        self.c_rep = new OpCRep_Clifford(<INT*>self.smatrix.data, <INT*>self.svector.data,
                                         <double complex*>self.unitary.data,
                                         <INT*>self.smatrix_inv.data, <INT*>self.svector_inv.data,
                                         <double complex*>self.unitary_dagger.data, self.state_space.num_qubits)

    def __reduce__(self):
        return (OpRepClifford, (self.unitary, (self.smatrix, self.svector), self.state_space))


cdef class OpRepStandard(OpRepClifford):   # TODO
    def __init__(self, name, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        super(OpRepStandard, self).__init__(U, None, state_space)


cdef class OpRepComposed(OpRep):
    cdef public object factor_reps  # list of OpRep objs?

    def __cinit__(self, factor_op_reps, state_space):
        self.factor_reps = factor_op_reps
        cdef INT i
        cdef INT nfactors = len(factor_op_reps)
        cdef vector[OpCRep*] creps = vector[OpCRep_ptr](nfactors)
        for i in range(nfactors):
            creps[i] = (<OpRep?>factor_op_reps[i]).c_rep
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Composed(creps, self.state_space.num_qubits)

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
        return OpRepComposed([f.copy() for f in self.factor_reps], self.state_space)


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
        self.c_rep = new OpCRep_Sum(factor_creps, self.state_space.num_qubits)

    def __reduce__(self):
        return (OpRepSum, (self.factor_reps, self.state_space))

    def copy(self):
        return OpRepSum([f.copy() for f in self.factor_reps], self.state_space)


cdef class OpRepEmbedded(OpRep):
    cdef public _np.ndarray qubits
    cdef public OpRep embedded_rep
    cdef public object state_space_labels
    cdef public object target_labels

    def __init__(self, state_space, target_labels, OpRep embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_tensor_product_blocks == 1
               and all([state_space.label_dimension(l) == 2 for l in state_space.tensor_product_block_labels(0)])), \
            "All state space labels must correspond to *qubits*"
        if isinstance(embedded_rep, OpRepClifford):
            assert(len(target_labels) == len(embedded_rep.svector) // 2), \
                "Inconsistent number of qubits in `target_labels` and `embedded_op`"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space.tensor_product_block_labels(0)
        cdef _np.ndarray[_np.int64_t, ndim=1, mode='c'] qubit_indices = \
            _np.array([qubitLabels.index(targetLbl) for targetLbl in target_labels], _np.int64)

        self.target_labels = target_labels
        self.qubits = qubit_indices
        self.embedded_rep = embedded_rep  # needed to prevent garbage collection?
        self.c_rep = new OpCRep_Embedded(embedded_rep.c_rep, state_space.num_qubits,
                                         <INT*>qubit_indices.data, <INT>qubit_indices.shape[0])
        self.state_space = state_space

    def __reduce__(self):
        return (OpRepEmbedded, (self.embedded_rep, self.state_space, self.target_labels))

    def copy(self):
        return OpRepEmbedded(self.embedded_rep.copy(), self.state_space, self.target_labels)


cdef class OpRepRepeated(OpRep):
    cdef public OpRep repeated_rep
    cdef public INT num_repetitions

    def __cinit__(self, OpRep rep_to_repeat, INT num_repetitions, state_space):
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        self.state_space = _StateSpace.cast(state_space)
        self.c_rep = new OpCRep_Repeated(self.repeated_rep.c_rep, num_repetitions, self.state_space.num_qubits)

    def __reduce__(self):
        return (OpRepRepeated, (self.repeated_rep, self.num_repetitions, self.state_space))

    def copy(self):
        return OpRepRepeated(self.repeated_rep.copy(), self.num_repetitions, self.state_space)


cdef class OpRepLindbladErrorgen(OpRep):
    def __init__(self, lindblad_term_dict, basis, state_space):
        super(OpRepLindbladErrorgen, self).__init__(state_space)
        self.Lterms = None
        self.Lterm_coeffs = None
        self.LtermdictAndBasis = (lindblad_term_dict, basis)
