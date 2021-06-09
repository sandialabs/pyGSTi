"""
State representations for "stabilizer_slow" evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace


def _update_chp_op(chp_op, old_to_new_qubit_index):
    if old_to_new_qubit_index is None:
        return chp_op
    else:
        return ''.join([(old_to_new_qubit_index[c] if c in old_to_new_qubit_index else c)
                        for c in chp_op])  # replaces, e.g. 'h 0' -> 'h 5'


class StateRep(_basereps.StateRep):
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

    def copy(self):
        return StateRep(self.chp_ops, self.state_space)


class StateRepComputational(StateRep):
    def __init__(self, zvals, basis, state_space):
        assert(all([x == 0 for x in zvals])), "Temporarily you can only specify the all-zeros state - TODO"
        self.basis = basis
        chp_ops = []
        super(StateRepComputational, self).__init__(chp_ops, state_space)


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__([], state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        self.chp_ops = self.state_rep.chp_ops + self.op_rep.chp_ops


class StateRepTensorProduct(StateRep):
    def __init__(self, factor_state_reps, state_space):
        self.factor_reps = factor_state_reps
        super(StateRepTensorProduct, self).__init__([], state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        chp_ops = []
        current_iqubit = 0
        for factor in self.factor_reps:
            local_to_tp_index = {str(iloc): str(itp) for iloc, itp in
                                 enumerate(range(current_iqubit, current_iqubit + factor.num_qubits))}
            chp_ops.extend([_update_chp_op(op, local_to_tp_index) for op in self.chp_ops])
            current_iqubit += factor.num_qubits
        self.chp_ops = chp_ops
