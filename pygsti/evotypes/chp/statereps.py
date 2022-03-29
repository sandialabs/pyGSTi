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
from pygsti.tools import internalgates as _itgs


def _update_chp_op(chp_op, old_to_new_qubit_index):
    if old_to_new_qubit_index is None:
        return chp_op
    else:
        new_op = chp_op.split()[0]
        for qubit in chp_op.split()[1:]:
            new_op += ' ' + old_to_new_qubit_index.get(qubit, qubit)
        return new_op


class StateRep(_basereps.StateRep):
    def __init__(self, chp_ops, state_space):
        self.base_chp_ops = chp_ops
        self.state_space = _StateSpace.cast(state_space)

        assert(self.state_space.num_qubits >= 0), 'State space for "chp" evotype must consist entirely of qubits!'
        assert(self.state_space.num_tensor_product_blocks == 1)  # should be redundant with above assertion
        self.qubit_labels = self.state_space.tensor_product_block_labels(0)
        self.qubit_label_to_index = {lbl: i for i, lbl in enumerate(self.qubit_labels)}

    @property
    def num_qubits(self):
        return self.state_space.num_qubits

    def chp_ops(self, seed_or_state=None):
        return self.base_chp_ops

    def chp_str(self, seed_or_state=None):
        op_str = '\n'.join(self.chp_ops(seed_or_state=seed_or_state))
        if len(op_str) > 0: op_str += '\n'
        return op_str

    def copy(self):
        return StateRep(self.base_chp_ops, self.state_space)


class StateRepComputational(StateRep):
    def __init__(self, zvals, basis, state_space):
        assert (basis.name in ['pp', 'PP']), "Only Pauli basis is allowed for 'chp' evotype"

        chp_ops = []
        x_ops = _itgs.standard_gatenames_chp_conversions()['Gxpi']
        for i, zval in enumerate(zvals):
            if zval:
                chp_ops.extend([_update_chp_op(x_op, {'0': str(i)}) for x_op in x_ops])
        
        super(StateRepComputational, self).__init__(chp_ops, state_space)


class StateRepComposed(StateRep):
    def __init__(self, state_rep, op_rep, state_space):
        self.state_rep = state_rep
        self.op_rep = op_rep
        super(StateRepComposed, self).__init__([], state_space)
        self.reps_have_changed()

    def reps_have_changed(self):
        self.base_chp_ops = self.state_rep.chp_ops() + self.op_rep.chp_ops()
    
    def chp_ops(self, seed_or_state=None):
        return self.state_rep.chp_ops(seed_or_state=seed_or_state) \
            + self.op_rep.chp_ops(seed_or_state=seed_or_state)

# TODO: Untested, only support computational and composed for now
#class StateRepTensorProduct(StateRep):
#    def __init__(self, factor_state_reps, state_space):
#        self.factor_reps = factor_state_reps
#        super(StateRepTensorProduct, self).__init__([], state_space)
#        self.reps_have_changed()
#
#    def reps_have_changed(self):
#        chp_ops = []
#        current_iqubit = 0
#        for factor in self.factor_reps:
#            local_to_tp_index = {str(iloc): str(itp) for iloc, itp in
#                                 enumerate(range(current_iqubit, current_iqubit + factor.num_qubits))}
#            chp_ops.extend([_update_chp_op(op, local_to_tp_index) for op in self.chp_ops])
#            current_iqubit += factor.num_qubits
#        self.chp_ops = chp_ops