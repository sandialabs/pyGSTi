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

import itertools as _itertools
import copy as _copy

import numpy as _np
from scipy.sparse.linalg import LinearOperator
from numpy.random import RandomState as _RandomState

from .statereps import StateRepDensePure as _StateRepDensePure
from .. import basereps as _basereps
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import basistools as _bt
from ...tools import internalgates as _itgs
from ...tools import optools as _ot


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
            return self.acton(in_state).to_dense('Hilbert')

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = _StateRepDensePure(_np.ascontiguousarray(v, complex), self.state_space, basis=None)
            return self.adjoint_acton(in_state).to_dense('Hilbert')
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

    def to_dense(self, on_space):
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


class OpRepStandard(OpRepDenseUnitary):
    def __init__(self, name, basis, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        state_space = _StateSpace.cast(state_space)
        assert(U.shape[0] == state_space.udim)

        super(OpRepStandard, self).__init__(U, basis, state_space)


#class OpRepStochastic(OpRepDense):
# - maybe we could add this, but it wouldn't be a "dense" op here,
#   perhaps we need to change API?


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

    def __init__(self, state_space, target_labels, embedded_rep):

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

        #dim = state_space.udim
        nBlocks = state_space.num_tensor_product_blocks
        iActiveBlock = iTensorProdBlk
        nComponents = len(state_space.tensor_product_block_labels(iActiveBlock))
        embeddedDim = embedded_rep.dim  # a *unitary* dim - see .dim property above
        blocksizes = _np.array([_np.prod(state_space.tensor_product_block_udimensions(k))
                                for k in range(nBlocks)], _np.int64)

        self.target_labels = target_labels
        self.embedded_rep = embedded_rep
        self.num_basis_els = numBasisEls
        self.action_inds = actionInds
        self.blocksizes = blocksizes

        num_basis_els_noop_blankaction = self.num_basis_els.copy()
        for i in self.action_inds: num_basis_els_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in num_basis_els_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(self.num_basis_els[1:]))))), _np.int64)
        self.basisInds_action = [list(range(self.num_basis_els[i])) for i in self.action_inds]

        self.embeddedDim = embeddedDim
        self.ncomponents = nComponents  # number of components in "active" block
        self.active_block_index = iActiveBlock
        self.nblocks = nBlocks
        self.offset = sum(blocksizes[0:self.active_block_index])
        super(OpRepEmbedded, self).__init__(state_space)

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.active_block_index:
                output_state.data[offset:offset + blockSize] = state.data[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = _StateRepDensePure(state.data[inds],
                                                  state.state_space.create_subspace(self.target_labels), basis=None)
            embedded_outstate = self.embedded_rep.acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = _StateRepDensePure(state.data[inds],
                                                  state.state_space.create_subspace(self.target_labels), basis=None)
            embedded_outstate = self.embedded_rep.adjoint_acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def acton_random(self, state, rand_state):
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = _StateRepDensePure(state.data[inds],
                                                  state.state_space.create_subspace(self.target_labels), basis=None)
            embedded_outstate = self.embedded_rep.acton_random(embedded_instate, rand_state)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton_random(self, state, rand_state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = _StateRepDensePure(_np.zeros(state.data.shape, complex), state.state_space, state.basis)
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = _StateRepDensePure(state.data[inds],
                                                  state.state_space.create_subspace(self.target_labels), basis=None)
            embedded_outstate = self.embedded_rep.adjoint_acton_random(embedded_instate, rand_state)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


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

    def to_dense(self, on_space):
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
