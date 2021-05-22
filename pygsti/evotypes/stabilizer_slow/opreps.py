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

from .. import basereps as _basereps
from ...tools import symplectic as _symp
from ...tools import matrixtools as _mt


class OpRep(_basereps.OpRep):
    def __init__(self, n):
        self.n = n  # number of qubits

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    @property
    def nqubits(self):
        return self.n

    @property
    def dim(self):
        return 2**(self.n)  # assume "unitary evolution"-type mode


class OpRepComposed(OpRep):

    def __init__(self, factor_op_reps, dim):
        n = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__(n)

    def reinit_factor_op_reps(self, factor_op_reps):
        self.factors_reps = factor_op_reps

    def acton(self, state):
        """ Act this gate map on an input state """
        for gate in self.factor_reps:
            state = gate.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factor_reps):
            state = gate.adjoint_acton(state)
        return state


class OpRepSum(OpRep):
    def __init__(self, factor_reps, dim):
        n = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
        self.factor_reps = factor_reps
        super(OpRepSum, self).__init__(n)

    def reinit_factor_reps(self, factor_reps):
        self.factors_reps = factor_reps

    def acton(self, state):
        """ Act this gate map on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        # need further stabilizer frame support to represent the sum of stabilizer states
        raise NotImplementedError()


class OpRepEmbedded(OpRep):

    def __init__(self, state_space_labels, target_labels, embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        assert(len(state_space_labels.labels) == 1
               and all([ld == 2 for ld in state_space_labels.labeldims.values()])), \
            "All state space labels must correspond to *qubits*"

        # Just a sanity check...  TODO REMOVE?
        #if isinstance(self.embedded_op, CliffordOp):
        #    assert(len(target_labels) == len(self.embedded_op.svector) // 2), \
        #        "Inconsistent number of qubits in `target_labels` and Clifford `embedded_op`"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space_labels.labels[0]
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        nQubits = state_space_labels.nqubits
        assert(nQubits is not None), "State space does not contain a definite number of qubits!"

        self.embedded_rep = embedded_rep
        self.qubits = qubit_indices  # qubit *indices*
        super(OpRepEmbedded, self).__init__(nQubits)

    def acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubits)
        outstate = self.embedded_rep.acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate

    def adjoint_acton(self, state):
        state = state.copy()  # needed?
        state.sframe.push_view(self.qubits)
        outstate = self.embedded_rep.adjoint_acton(state)  # works b/c sfame has "view filters"
        state.sframe.pop_view()  # return input state to original view
        outstate.sframe.pop_view()
        return outstate


class OpRepClifford(OpRep):
    def __init__(self, unitarymx, symplecticrep):

        if symplecticrep is not None:
            self.smatrix, self.svector = symplecticrep
        else:
            # compute symplectic rep from unitary
            self.smatrix, self.svector = _symp.unitary_to_symplectic(unitarymx, flagnonclifford=True)

        self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
            self.smatrix, self.svector)  # cache inverse since it's expensive

        #nQubits = len(self.svector) // 2
        #dim = 2**nQubits  # "stabilizer" is a "unitary evolution"-type mode
        self.unitary = unitarymx

    @property
    def unitary_dagger(self):
        return _np.conjugate(self.unitary.T)

    def acton(self, state):
        """ Act this gate map on an input state """
        state = state.copy()  # (copies any qubit filters in .sframe too)
        state.sframe.clifford_update(self.smatrix, self.svector, self.unitary)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        # Note: cliffords are unitary, so adjoint == inverse
        state = state.copy()  # (copies any qubit filters in .sframe too)
        state.sframe.clifford_update(self.smatrix_inv, self.svector_inv,
                                     _np.conjugate(self.unitary.T))
        return state

    def __str__(self):
        """ Return string representation """
        s = "Clifford operation with matrix:\n"
        s += _mt.mx_to_string(self.smatrix, width=2, prec=0)
        s += " and vector " + _mt.mx_to_string(self.svector, width=2, prec=0)
        return s


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, dim):
        nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(nQubits)

    def acton(self, state):
        """ Act this gate map on an input state """
        for i in range(self.num_repetitions):
            state = self.repeated_rep.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.num_repetitions):
            state = self.repated_rep.adjoint_acton(state)
        return state
