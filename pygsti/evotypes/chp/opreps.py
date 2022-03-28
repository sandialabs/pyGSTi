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
from numpy.random import RandomState as _RandomState

from .statereps import _update_chp_op
from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs


class OpRep(_basereps.OpRep):
    # Currently the same as StateRep... combine somehow?
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
            "State space of {0} qubits doesn't match {1} expected qubits for the standard {2} gate".format(
                state_space.num_qubits, nqubits, name)

        self.basis = basis
        super(OpRepStandard, self).__init__(chp_ops, state_space)


class OpRepComposed(OpRep):

    def __init__(self, factor_op_reps, state_space):
        state_space = _StateSpace.cast(state_space)
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__([], state_space)

    def reinit_factor_op_reps(self, factor_op_reps):
        self.factors_reps = factor_op_reps

    def chp_ops(self, seed_or_state=None):
        ops = []
        for factor in self.factor_reps:
            ops.extend(factor.chp_ops)

        return ops


class OpRepEmbedded(OpRep):

    def __init__(self, state_space, target_labels, embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_tensor_product_blocks == 1
               and all([state_space.label_udimension(l) == 2 for l in state_space.tensor_product_block_labels(0)])), \
            "All state space labels must correspond to *qubits*"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space.tensor_product_block_labels(0)
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        self.embedded_labels = target_labels
        self.embedded_rep = embedded_rep
        # Map 0-based qubit index for embedded op -> full local qubit index
        self.embedded_to_local_qubit_indices = {str(i): str(j) for i, j in enumerate(qubit_indices)}

        # TODO: This doesn't work as nicely for the stochastic op, where chp_ops can be reset between chp_str calls
        chp_ops = [_update_chp_op(op, self.embedded_to_local_qubit_indices) for op in self.embedded_rep.chp_ops()]
        super(OpRepEmbedded, self).__init__(chp_ops, state_space)

    @property
    def chp_ops(self, seed_or_state=None):
        return [_update_chp_op(op, self.embedded_to_local_qubit_indices) for op in self.embedded_rep.chp_ops(seed_or_state=seed_or_state)]


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, state_space):
        state_space = _StateSpace.cast(state_space)
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(self.repeated_rep.chp_ops() * self.num_repetitions, state_space)


class OpRepStochastic(OpRep):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state, state_space):

        self.basis = basis
        assert (basis.name in ['pp', 'PP']), "Only Pauli basis is allowed for 'chp' evotype"

        if isinstance(seed_or_state, _RandomState):
            self.rand_state = seed_or_state
        else:
            self.rand_state = _RandomState(seed_or_state)

        #TODO: need to fix this: `basis` above functions as basis to make superoperators out of, but here we have
        # a CHP stochastic op which is given a basis for the space - e.g. a dim=2 vector space for 1 qubit, so
        # we need to distinguish/specify the basis better for this... and what about rate_poly_dicts (see svterm)
        nqubits = state_space.num_qubits
        assert(self.basis.dim == 4**nqubits), "Must have an integral number of qubits"

        std_chp_ops = _itgs.standard_gatenames_chp_conversions()

        # For CHP, need to make a Composed + EmbeddedOp for the super operators
        # For lower overhead, make this directly using the rep instead of with objects
        self.stochastic_superop_reps = []
        for label in self.basis.labels[1:]:
            combined_chp_ops = []

            for i, pauli in enumerate(label):
                name = 'Gi' if pauli == "I" else 'G%spi' % pauli.lower()
                chp_op = std_chp_ops[name]
                chp_op_targeted = [op.replace('0', str(i)) for op in chp_op]
                combined_chp_ops.extend(chp_op_targeted)

            sub_rep = OpRep(combined_chp_ops, state_space)
            self.stochastic_superop_reps.append(sub_rep)
        self.rates = initial_rates
        super(OpRepStochastic, self).__init__([], state_space)  # don't store any chp_ops in base

    def update_rates(self, rates):
        self.rates[:] = rates

    def chp_ops(self, seed_or_state=None):
        # Optionally override RNG for this call
        if seed_or_state is not None:
            if isinstance(seed_or_state, _np.random.RandomState):
                rand_state = seed_or_state
            else:
                rand_state = _np.random.RandomState(seed_or_state)
        else:
            rand_state = self.rand_state
        
        rates = self.rates
        all_rates = [*rates, 1.0 - sum(rates)]  # Include identity so that probabilities are 1
        index = rand_state.choice(self.basis.size, p=all_rates)

        # If final entry, no operation selected
        if index == self.basis.size - 1:
            return ''

        rep = self.stochastic_superop_reps[index]
        return rep.chp_ops
