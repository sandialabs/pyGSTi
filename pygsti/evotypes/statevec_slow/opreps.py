"""
Operation representation classes for the `statevec_slow` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from collections.abc import Sequence as _Sequence
import copy as _copy

import numpy as _np
from scipy.sparse.linalg import LinearOperator
from numpy.random import RandomState as _RandomState

from .statereps import StateRep as _StateRep, StateRepDensePure as _StateRepDensePure
from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import internalgates as _itgs
from ...tools import optools as _ot
from pygsti import SpaceT

class OpRep(_basereps.OpRep):
    def __init__(self, state_space):
        self.state_space = state_space

    @property
    def dim(self):
        return self.state_space.udim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    def acton_random(self, state, rand_state):
        return self.acton(state)  # default is to ignore rand_state

    def adjoint_acton_random(self, state, rand_state):
        return self.adjoint_acton(state)  # default is to ignore rand_state

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = _StateRepDensePure(_np.ascontiguousarray(v, complex), self.state_space, basis=None)
            return self.acton(in_state).to_dense("Hilbert")

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = _StateRepDensePure(_np.ascontiguousarray(v, complex), self.state_space, basis=None)
            return self.adjoint_acton(in_state).to_dense("Hilbert")
        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?

    def copy(self):
        return _copy.deepcopy(self)


class OpRepDenseUnitary(OpRep):
    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.udim, complex)
        assert(mx.ndim == 2 and mx.shape[0] == state_space.udim)
        self.basis = basis
        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        super(OpRepDenseUnitary, self).__init__(state_space)

    def base_has_changed(self):
        pass

    def to_dense(self, on_space: SpaceT = 'minimal'):
        if on_space in ('minimal', 'Hilbert'):
            return self.base
        elif on_space == 'HilbertSchmidt':
            return _ot.unitary_to_superop(self.base, self.basis)
        else:
            raise ValueError("Invalid `on_space` argument: %s" % str(on_space))

    def acton(self, state):
        return _StateRepDensePure(_np.dot(self.base, state.data), state.state_space, state.basis)

    def adjoint_acton(self, state):
        return _StateRepDensePure(_np.dot(_np.conjugate(self.base.T), state.data),
                                  state.state_space, state.basis)

    def __str__(self):
        return "OpRepDenseUnitary:\n" + str(self.base)


class OpRepComposed(OpRep):
    # exactly the same as densitymx case
    def __init__(self, factor_op_reps, state_space):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_op_reps
        super(OpRepComposed, self).__init__(state_space)

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

    def acton_random(self, state, rand_state):
        """ Act this gate map on an input state """
        for gate in self.factor_reps:
            state = gate.acton_random(state, rand_state)
        return state

    def adjoint_acton_random(self, state, rand_state):
        """ Act the adjoint of this operation matrix on an input state """
        for gate in reversed(self.factor_reps):
            state = gate.adjoint_acton_random(state, rand_state)
        return state

    def reinit_factor_op_reps(self, new_factor_op_reps):
        self.factors_reps = new_factor_op_reps


class OpRepSum(OpRep):
    # exactly the same as densitymx case
    def __init__(self, factor_reps, state_space):
        #assert(len(factor_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(OpRepSum, self).__init__(state_space)

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        for f in self.factor_reps:
            output_state.data += f.acton(state).data
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        for f in self.factor_reps:
            output_state.data += f.adjoint_acton(state).data
        return output_state

    def acton_random(self, state, rand_state):
        """ Act this gate map on an input state """
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        for f in self.factor_reps:
            output_state.data += f.acton_random(state, rand_state).data
        return output_state

    def adjoint_acton_random(self, state, rand_state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        for f in self.factor_reps:
            output_state.data += f.adjoint_acton_random(state, rand_state).data
        return output_state


class OpRepEmbedded(OpRep):

    def __init__(self, state_space: _StateSpace.Castable, target_labels: _Sequence[str | int],
                 embedded_rep: OpRep) -> None:

        state_space = _StateSpace.cast(state_space)
        iTensorProdBlks = [state_space.label_tensor_product_block_index(label) for label in target_labels]
        # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit operation must correspond to the"
                             " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

        iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
        tensorProdBlkLabels = state_space.tensor_product_block_labels(iTensorProdBlk)
        # count possible *state-vector-space* indices of each component of the tensor product block
        numBasisEls = _np.array([state_space.label_udimension(l) for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        actionInds = _np.array(labelIndices, _np.int64)
        assert(_np.prod([numBasisEls[i] for i in actionInds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        blocksizes = [int(_np.prod(dims)) for dims in state_space.tensor_product_blocks_udimensions]
        offset = sum(blocksizes[:iTensorProdBlk])
        multipliers = _np.append(_np.cumprod(numBasisEls[:0:-1])[::-1], 1).astype(_np.int64)

        def _flat_offsets(label_inds: _Sequence[int]) -> _np.ndarray:
            offsets = _np.zeros(1, _np.int64)
            for i in label_inds:
                offsets = (offsets[:, None] + multipliers[i] * _np.arange(numBasisEls[i])[None, :]).ravel()
            return offsets

        untouched = [i for i in range(len(numBasisEls)) if i not in labelIndices]
        self.embedded_rep = embedded_rep
        self._embedded_state_space = state_space.create_subspace(target_labels)
        self._block_base_inds = offset + _flat_offsets(untouched)
        self._embedded_rel_inds = _flat_offsets(labelIndices)
        super(OpRepEmbedded, self).__init__(state_space)

    def _apply(self, state: _StateRep, adjoint: bool) -> _StateRepDensePure:
        # Read the current matrix on every action: embedded parameters may have changed.
        if isinstance(self.embedded_rep, OpRepDenseUnitary):
            mx = self.embedded_rep.to_dense('Hilbert')
            if adjoint:
                mx = mx.conj().T
        else:
            # Composed and other structured reps need not expose a dense matrix.
            linop = self.embedded_rep.aslinearoperator()
            identity = _np.identity(self.embedded_rep.dim, complex)
            mx = linop.rmatmat(identity) if adjoint else linop.matmat(identity)
        inds = self._block_base_inds[:, None] + self._embedded_rel_inds[None, :]
        out = state.data.copy()
        out[inds] = state.data[inds] @ mx.T
        return _StateRepDensePure(out, state.state_space, state.basis)

    def acton(self, state: _StateRep) -> _StateRepDensePure:
        return self._apply(state, False)

    def adjoint_acton(self, state: _StateRep) -> _StateRepDensePure:
        return self._apply(state, True)

    def _apply_random(self, state: _StateRep, rand_state: _RandomState | _np.random.Generator | None,
                      adjoint: bool) -> _StateRepDensePure:
        # Preserve one random draw per untouched-index block, in its original order.
        action = self.embedded_rep.adjoint_acton_random if adjoint else self.embedded_rep.acton_random
        out = state.data.copy()
        for base in self._block_base_inds:
            inds = base + self._embedded_rel_inds
            substate = _StateRepDensePure(state.data[inds], self._embedded_state_space, basis=None)
            out[inds] = action(substate, rand_state).data
        return _StateRepDensePure(out, state.state_space, state.basis)

    def acton_random(self, state: _StateRep,
                     rand_state: _RandomState | _np.random.Generator | None) -> _StateRepDensePure:
        return self._apply_random(state, rand_state, False)

    def adjoint_acton_random(self, state: _StateRep,
                             rand_state: _RandomState | _np.random.Generator | None) -> _StateRepDensePure:
        return self._apply_random(state, rand_state, True)


class OpRepExpErrorgen(OpRep):
    """
    Representation of an operation that is the exponential of an error generator.

    Note: in this evotype this representation cannot act on states (`acton` raises); it exists so
    that Lindblad error-generator-based operations can be constructed with this evotype.  Such
    operations can be simulated only via the term-based machinery of the deprecated
    `TermForwardSimulator` (see #916), but they may still be constructed for other uses
    (e.g. models built for error generator propagation), so this class is *not* deprecated.
    The term-related attributes (e.g. `Lterms`, `Lterm_coeffs`) are used only by the
    deprecated term machinery.
    """

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

    def acton_random(self, state, rand_state):
        """ Act this gate map on an input state """
        for i in range(self.num_repetitions):
            state = self.repeated_rep.acton_random(state, rand_state)
        return state

    def adjoint_acton_random(self, state, rand_state):
        """ Act the adjoint of this operation matrix on an input state """
        for i in range(self.num_repetitions):
            state = self.repeated_rep.adjoint_acton_random(state, rand_state)
        return state


class OpRepLindbladErrorgen(OpRep):
    """
    Representation of a Lindblad error generator (defined by Lindblad coefficient blocks).

    Note: in this evotype this representation cannot act on states (`acton` raises); it exists so
    that Lindblad error-generator-based operations can be constructed with this evotype.  Such
    operations can be simulated only via the term-based machinery of the deprecated
    `TermForwardSimulator` (see #916), but they may still be constructed for other uses
    (e.g. models built for error generator propagation), so this class is *not* deprecated.
    The term-related attributes (e.g. `Lterms`, `Lterm_coeffs`) are used only by the
    deprecated term machinery.
    """
    def __init__(self, lindblad_coefficient_blocks, state_space):
        super(OpRepLindbladErrorgen, self).__init__(state_space)
        self.Lterms = None
        self.Lterm_coeffs = None
        self.lindblad_coefficient_blocks = lindblad_coefficient_blocks


class OpRepRandomUnitary(OpRep):
    def __init__(self, basis, unitary_rates, unitary_reps, seed_or_state, state_space):
        self.basis = basis
        self.unitary_reps = unitary_reps
        self.unitary_rates = unitary_rates.copy()

        if isinstance(seed_or_state, _RandomState):
            self.rand_state = seed_or_state
        else:
            self.rand_state = _RandomState(seed_or_state)

        self.state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == self.state_space.dim)
        super(OpRepRandomUnitary, self).__init__(state_space)

    def acton_random(self, state, rand_state):
        rand_state = rand_state if rand_state is not None else self.rand_state
        rates = self.unitary_rates
        index = rand_state.choice(len(self.unitary_rates), p=rates)
        rep = self.unitary_reps[index]
        return rep.acton(state)

    def adjoint_acton_random(self, state, rand_state):
        rand_state = rand_state if rand_state is not None else self.rand_state
        rates = self.unitary_rates
        index = rand_state.choice(len(self.unitary_rates), p=rates)
        rep = self.unitary_reps[index]
        return rep.adjoint_acton(state)

    def __str__(self):
        return "OpRepRandomUnitary:\n" + " rates: " + str(self.unitary_rates)  # maybe show ops too?

    def copy(self):
        return OpRepRandomUnitary(self.basis, self.unitary_rates, list(self.unitary_reps),
                                  self.rand_state, self.state_space)

    def update_unitary_rates(self, rates):
        self.unitary_rates[:] = rates

    def to_dense(self, on_space: SpaceT = 'minimal'):
        assert(on_space == 'HilbertSchmidt')  # below code only works in this case
        return sum([rate * rep.to_dense(on_space) for rate, rep in zip(self.unitary_rates, self.unitary_reps)])


class OpRepStochastic(OpRepRandomUnitary):

    def __init__(self, stochastic_basis, basis, initial_rates, seed_or_state, state_space):
        self.rates = initial_rates
        self.stochastic_basis = stochastic_basis
        rates = [1 - sum(initial_rates)] + list(initial_rates)
        reps = [OpRepDenseUnitary(bel, basis, state_space) for bel in stochastic_basis.elements]
        assert(len(reps) == len(rates))

        state_space = _StateSpace.cast(state_space)
        assert(basis.dim == state_space.dim)

        super(OpRepStochastic, self).__init__(basis, _np.array(rates, 'd'), reps, seed_or_state, state_space)

    def update_rates(self, rates):
        unitary_rates = [1 - sum(rates)] + list(rates)
        self.rates[:] = rates
        self.update_unitary_rates(unitary_rates)
