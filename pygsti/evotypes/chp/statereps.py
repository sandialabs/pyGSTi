"""
State representations for "stabilizer_slow" evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
        self.chp_ops = chp_ops
        self.state_space = _StateSpace.cast(state_space)

        assert(self.state_space.num_qubits >= 0), 'State space for "chp" evotype must consist entirely of qubits!'
        assert(self.state_space.num_tensor_product_blocks == 1)  # should be redundant with above assertion
        self.qubit_labels = self.state_space.sole_tensor_product_block_labels
        self.qubit_label_to_index = {lbl: i for i, lbl in enumerate(self.qubit_labels)}

    @property
    def num_qubits(self):
        return self.state_space.num_qubits

    def copy(self):
        return StateRep(self.chp_ops, self.state_space)

    def actionable_staterep(self):
        # return a state rep that can be acted on by op reps or mapped to
        # a probability/amplitude by POVM effect reps.
        return self  # for most classes, the rep itself is actionable


class StateRepComputational(StateRep):
    def __init__(self, zvals, basis, state_space):
        assert all([nm in ('pp', 'PP') for nm in basis.name.split('*')]), \
            "Only Pauli basis is allowed for 'chp' evotype"

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
        #REMOVE self.reps_have_changed()

    def reps_have_changed(self):
        pass  # not needed -- don't actually hold ops
        #REMOVE self.base_chp_ops = self.state_rep.chp_ops() + self.op_rep.chp_ops()

    def actionable_staterep(self):
        state_rep = self.state_rep.actionable_staterep()
        return self.op_rep.acton(state_rep)
