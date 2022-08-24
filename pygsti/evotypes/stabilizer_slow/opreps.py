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
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs
from ...tools import matrixtools as _mt
from ...tools import symplectic as _symp


class OpRep(_basereps.OpRep):
    def __init__(self, state_space):
        assert(state_space.num_qubits is not None), "State space does not contain a definite number of qubits!"
        self.state_space = state_space

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    @property
    def nqubits(self):
        return self.state_space.num_qubits

    #@property
    #def dim(self):
    #    return 2**(self.nqubits)  # assume "unitary evolution"-type mode


class OpRepClifford(OpRep):
    def __init__(self, unitarymx, symplecticrep, basis, state_space):

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
        self.basis = basis

        state_space = _StateSpace.cast(state_space)
        assert(state_space.num_qubits == self.smatrix.shape[0] // 2)
        super(OpRepClifford, self).__init__(state_space)

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


class OpRepStandard(OpRepClifford):
    def __init__(self, name, basis, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        super(OpRepStandard, self).__init__(U, None, basis, state_space)


class OpRepComposed(OpRep):

    def __init__(self, factor_op_reps, state_space):
        state_space = _StateSpace.cast(state_space)
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__(state_space)

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
    def __init__(self, factor_reps, state_space):
        state_space = _StateSpace.cast(state_space)
        self.factor_reps = factor_reps
        super(OpRepSum, self).__init__(state_space)

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

    def __init__(self, state_space, target_labels, embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        state_space = _StateSpace.cast(state_space)
        assert(all([state_space.label_udimension(l) == 2 for l in state_space.sole_tensor_product_block_labels])), \
            "All state space labels must correspond to *qubits*"

        #Cache info to speedup representation's acton(...) methods:
        # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
        qubitLabels = state_space.sole_tensor_product_block_labels
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        self.embedded_rep = embedded_rep
        self.qubits = qubit_indices  # qubit *indices*
        super(OpRepEmbedded, self).__init__(state_space)

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


class OpRepExpErrorgen(OpRep):

    def __init__(self, errorgen_rep):
        state_space = errorgen_rep.state_space
        self.errorgen_rep = errorgen_rep
        super(OpRepExpErrorgen, self).__init__(state_space)

    def errgenrep_has_changed(self, onenorm_upperbound):
        pass

    def acton(self, state):
        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")

    def adjoint_acton(self, state):
        raise AttributeError("Cannot currently act with statevec.OpRepExpErrorgen - for terms only!")


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, state_space):
        state_space = _StateSpace.cast(state_space)
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(state_space)

    def acton(self, state):
        """ Act this gate map on an input state """
        for i in range(self.num_repetitions):
            state = self.repeated_rep.acton(state)
        return state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.num_repetitions):
            state = self.repeated_rep.adjoint_acton(state)
        return state


class OpRepLindbladErrorgen(OpRep):
    def __init__(self, lindblad_coefficient_blocks, state_space):
        super(OpRepLindbladErrorgen, self).__init__(state_space)
        self.Lterms = None
        self.Lterm_coeffs = None
        self.lindblad_coefficient_blocks = lindblad_coefficient_blocks
