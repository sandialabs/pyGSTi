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
        super(OpRepComposed, self).__init__(state_space)

    def reinit_factor_op_reps(self, factor_op_reps):
        self.factors_reps = factor_op_reps

    def chp_str(self, target_labels=None):
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


class OpRepStochastic(OpRep):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state, state_space):

        self.basis = basis
        assert (basis.name == 'pp'), "Only Pauli basis is allowed for 'chp' evotype"

        if isinstance(seed_or_state, _RandomState):
            self.rand_state = seed_or_state
        else:
            self.rand_state = _RandomState(seed_or_state)

        #TODO: need to fix this: `basis` above functions as basis to make superoperators out of, but here we have
        # a CHP stochastic op which is given a basis for the space - e.g. a dim=2 vector space for 1 qubit, so
        # we need to distinguish/specify the basis better for this... and what about rate_poly_dicts (see svterm)
        nqubits = state_space.num_qubits  # OLD REMOVE: (self.basis.dim - 1).bit_length()
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

    def chp_str(self, target_labels=None):
        """Return a string suitable for printing to a CHP input file after stochastically selecting operation.

        Parameters
        ----------
        targets: list of int
            Qubits to be applied to (if None, uses stored CHP strings directly)

        Returns
        -------
        s : str
            String of CHP code
        """
        assert (self._evotype == 'chp'), "Must have 'chp' evotype to use get_chp_str"

        rates = self.rates
        all_rates = [*rates, 1.0 - sum(rates)]  # Include identity so that probabilities are 1
        index = self.rand_state.choice(self.basis.size, p=all_rates)

        # If final entry, no operation selected
        if index == self.basis.size - 1:
            return ''

        rep = self.stochastic_superop_reps[index]
        self.chp_ops = rep.chp_ops  # set our chp_ops so call to base-class property below uses these
        return OpRep.chp_str.fget(self)


### REMOVE - SCRATH for updating the above:
#class OpRepComposed(XXX):
#
#    def __init__(self, factor_op_reps, dim):
#        pass
#
#    def reinit_factor_op_reps(factor_op_reps):
#        pass  # TODO
#
#    def get_chp_str(self, targets=None):
#        """Return a string suitable for printing to a CHP input file from all underlying operations.
#
#        Parameters
#        ----------
#        targets: list of int
#            Qubits to be applied to (if None, uses stored CHP strings directly)
#
#        Returns
#        -------
#        s : str
#            String of CHP code
#        """
#        s = ""
#        for op in self.factorops:
#            s += op.get_chp_str(targets)
#        return s
#
#
#class OpRepEmbedded(XXX):
#
#    def __init__(self, state_space_labels, target_labels, embedded_rep):
#        # assert that all state space labels == qubits, since we only know
#        # how to embed cliffords on qubits...
#        assert(len(state_space_labels.labels) == 1
#               and all([ld == 2 for ld in state_space_labels.labeldims.values()])), \
#            "All state space labels must correspond to *qubits*"
#
#        #TODO: enfore this another way?
#        #assert(self.embedded_op._evotype == 'chp'), \
#        #    "Embedded op must also have CHP evotype instead of %s" % self.embedded_op._evotype
#        assert(isinstance(embedded_rep, OpRepBase))  # needs to point to chp.OpRep class??
#
#        op_nqubits = (embedded_rep.dim - 1).bit_length()
#        assert(len(target_labels) == op_nqubits), \
#            "Inconsistent number of qubits in `target_labels` ({0}) and CHP `embedded_op` ({1})".format(
#                len(target_labels), op_nqubits)
#
#        qubitLabels = state_space_labels.labels[0]
#        qubit_indices = _np.array([qubitLabels.index(targetLbl)
#                                   for targetLbl in target_labels], _np.int64)
#
#        nQubits = state_space_labels.nqubits
#        assert(nQubits is not None), "State space does not contain a definite number of qubits!"
#
#        # Store qubit indices as targets for later use
#        self.target_indices = qubit_indices
#
#        #TODO - figure out what this means - I think there wasn't a CHP embedded rep class before?
#        rep = opDim  # Don't set representation again, just use embedded_op calls later
#
#
#    def get_chp_str(self, targets=None):  # => chpstr property? TODO
#        """Return a string suitable for printing to a CHP input file from the embedded operations.
#
#        Just calls underlying get_chp_str but with an extra layer of target redirection.
#
#        Parameters
#        ----------
#        targets: list of int
#            Qubits to be applied to (if None, uses stored CHP strings directly).
#
#        Returns
#        -------
#        s : str
#            String of CHP code
#        """
#        target_indices = list(self.target_indices)
#
#        # Targets are for the full embedded operation so we need to map these to the actual targets of the CHP op
#        if targets is not None:
#            assert len(targets) == self.state_space_labels.nqubits, \
#                "Got {0} targets instead of required {1}".format(len(targets), self.state_space_labels.nqubits)
#            target_indices = [targets[ti] for ti in self.target_indices]
#
#        return self.embedded_op.get_chp_str(target_indices)
#
#
#class OpRepStandard(XXX):
#    def __init__(self, name):
#        self.name = name
#        std_chp_ops = _itgs.standard_gatenames_chp_conversions()
#        if self.name not in std_chp_ops:
#            raise ValueError("Name '%s' not in standard CHP operations" % self.name)
#
#        native_ops = std_chp_ops[self.name]
#        nqubits = 2 if any(['c' in n for n in native_ops]) else 1
#
#        rep = replib.CHPOpRep(native_ops, nqubits)
