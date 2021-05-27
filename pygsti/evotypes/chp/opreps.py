"""
Operation representation classes for the `stabilizer_slow` evolution type.
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

from .statereps import _update_chp_op
from .. import basereps as _basereps
from ...models.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs


class OpRep(_basereps.OpRep):
    # Currently the same as StateRep... combine somehow?
    def __init__(self, chp_ops, state_space):
        self.chp_ops = chp_ops
        self.state_space = _StateSpace.cast(state_space)

        assert(self.state_space.num_qubits >= 0), 'State space for "chp" evotype must consist entirely of qubits!'
        assert(self.state_space.num_tensor_product_blocks == 1)  # should be redundant with above assertion
        self.qubit_labels = self.state_space.tensor_product_block_labels(0)
        self.qubit_label_to_index = {lbl: i for i, lbl in enumerate(self.qubit_labels)}

    @property
    def num_qubits(self):
        return self.state_space.num_qubits

    def chp_str(self, target_labels=None):
        if target_labels is not None:
            assert(len(target_labels) == self.num_qubits), \
                "Got {0} target labels instead of required {1}".format(len(target_labels), self.num_qubits)

            # maps 0-based local op qubit index -> global qubit index
            global_target_index = {str(i): str(self.qubit_label_to_index[lbl]) for i, lbl in enumerate(target_labels)}
            op_str = '\n'.join(map(lambda op: _update_chp_op(op, global_target_index), self.chp_ops))
        else:
            op_str = '\n'.join(self.chp_ops)
        return op_str


class OpRepClifford(OpRep):
    def __init__(self, unitarymx, symplecticrep, basis, state_space):

        raise NotImplementedError(("This could be implemented in the future - we just need"
                                   "to decompose an arbitrary Clifford unitary/stabilizer into CHP ops"))
        chp_ops = []  # compile_clifford_unitary_to_chp(unitarymx) TODO!!!
        state_space = _StateSpace.cast(state_space)
        self.basis = basis
        super(OpRepClifford, self).__init__(chp_ops, state_space)


class OpRepStandard(OpRep):
    def __init__(self, name, basis, state_space):
        std_chp_ops = _itgs.standard_gatenames_chp_conversions()
        self.name = name
        if self.name not in std_chp_ops:
            raise ValueError("Name '%s' not in standard CHP operations" % self.name)

        chp_ops = std_chp_ops[self.name]
        nqubits = 2 if any(['c' in n for n in chp_ops]) else 1

        state_space = _StateSpace.cast(state_space)
        assert(nqubits == state_space.num_qubits), \
            "State space of {0} qubits doesn't match {1} expected qubits for the standard {1} gate".format(
                state_space.num_qubits, nqubits, name)

        self.basis = basis
        super(OpRepStandard, self).__init__(chp_ops, state_space)


class OpRepComposed(OpRep):

    def __init__(self, factor_op_reps, state_space):
        state_space = _StateSpace.cast(state_space)
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__(state_space)

    def reinit_factor_op_reps(self, factor_op_reps):
        self.factors_reps = factor_op_reps

    def chp_str(self, target_labels):
        return ''.join([factor.chp_str(target_labels) for factor in self.factor_reps])


class OpRepEmbedded(OpRep):

    def __init__(self, state_space, target_labels, embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_tensor_product_blocks == 1
               and all([state_space.label_dimension(l) == 2 for l in state_space.tensor_product_block_labels(0)])), \
            "All state space labels must correspond to *qubits*"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space.tensor_product_block_labels(0)
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        self.target_labels = target_labels
        self.embedded_rep = embedded_rep
        embedded_to_local_qubit_indices = {str(i): str(j) for i, j in enumerate(qubit_indices)}

        chp_ops = [_update_chp_op(op, embedded_to_local_qubit_indices) for op in self.embedded_rep.chp_ops]
        super(OpRepEmbedded, self).__init__(chp_ops, state_space)


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, state_space):
        state_space = _StateSpace.cast(state_space)
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(self.repeated_rep.chp_ops * self.num_repetitions, state_space)
