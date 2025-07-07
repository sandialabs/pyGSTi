"""
Operation representation classes for the `densitymx_slow` evolution type.
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

import numpy as _np
import scipy.sparse as _sps
from scipy.sparse.linalg import LinearOperator

from .statereps import StateRepDense as _StateRepDense
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from ...tools import basistools as _bt
from ...tools import internalgates as _itgs
from ...tools import lindbladtools as _lbt
from ...tools import matrixtools as _mt
from ...tools import optools as _ot
from pygsti.enums import SpaceConversionType

class OpRep:
    """
    A real superoperator on Hilbert-Schmidt space.
    """

    def __init__(self, state_space):
        self.state_space = state_space

    @property
    def dim(self):
        return self.state_space.dim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    def aslinearoperator(self):
        """
        Return a SciPy LinearOperator that accepts superket representations of vectors
        in Hilbert-Schmidt space and returns a vector of that same representation.
        """
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = _StateRepDense(_np.ascontiguousarray(v, 'd'), self.state_space, None)
            return self.acton(in_state).to_dense(on_space=SpaceConversionType.HilbertSchmidt)

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = _StateRepDense(_np.ascontiguousarray(v, 'd'), self.state_space, None)
            return self.adjoint_acton(in_state).to_dense(on_space=SpaceConversionType.HilbertSchmidt)
        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?


class OpRepDenseSuperop(OpRep):
    """
    A real superoperator on Hilbert-Schmidt space.
    The operator's action (and adjoint action) work with Hermitian matrices
    stored as *vectors* in their real superket representations.
    """

    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.dim, 'd')
        assert(mx.ndim == 2 and mx.shape[0] == state_space.dim)

        self.basis = basis
        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        super(OpRepDenseSuperop, self).__init__(state_space)

    def base_has_changed(self):
        pass

    def to_dense(self, on_space):
        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx_slow' evotype cannot produce Hilbert-space ops!")
        return self.base

    def acton(self, state):
        return _StateRepDense(_np.dot(self.base, state.data), state.state_space, None)  # state.basis if it had one

    def adjoint_acton(self, state):
        return _StateRepDense(_np.dot(self.base.T, state.data), state.state_space, None)  # no conjugate b/c *real* data

    def __str__(self):
        return "OpRepDenseSuperop:\n" + str(self.base)

    def copy(self):
        return OpRepDenseSuperop(self.base.copy(), self.basis, self.state_space)


class OpRepDenseUnitary(OpRep):

    def __init__(self, mx, basis, state_space):
        state_space = _StateSpace.cast(state_space)
        if mx is None:
            mx = _np.identity(state_space.dim, 'd')
        assert(mx.ndim == 2 and mx.shape[0] == state_space.udim)

        self.basis = basis
        self.base = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.superop_base = _np.require(_ot.unitary_to_superop(mx, self.basis),
                                        requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.state_space = state_space

    def base_has_changed(self):
        self.superop_base[:, :] = _ot.unitary_to_superop(self.base, self.basis)

    def to_dense(self, on_space):
        if on_space in ('minimal', 'HilbertSchmidt'):
            return self.to_dense_superop()
        else:  # 'Hilbert'
            return self.base

    def to_dense_superop(self):
        return self.superop_base

    def acton(self, state):
        return _StateRepDense(_np.dot(self.superop_base, state.data), state.state_space, None)

    def adjoint_acton(self, state):
        return _StateRepDense(_np.dot(self.superop_base.T, state.data), state.state_space, None)
        # no conjugate b/c *real* data

    def __str__(self):
        return "OpRepDenseUnitary:\n" + str(self.base)

    def copy(self):
        return OpRepDenseUnitary(self.base.copy(), self.basis, self.state_space)


class OpRepSparse(OpRep):
    def __init__(self, a_data, a_indices, a_indptr, state_space):
        dim = len(a_indptr) - 1
        state_space = _StateSpace.cast(state_space)
        assert(state_space.dim == dim)
        self.A = _sps.csr_matrix((a_data, a_indices, a_indptr), shape=(dim, dim))
        super(OpRepSparse, self).__init__(state_space)

    @property
    def data(self):
        return self.A.data

    @property
    def indices(self):
        return self.A.indices

    @property
    def indptr(self):
        return self.A.indptr

    def acton(self, state):
        """ Act this gate map on an input state """
        return _StateRepDense(self.A.dot(state.data), state.state_space, None)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        Aadj = self.A.conjugate(copy=True).transpose()
        return _StateRepDense(Aadj.dot(state.data), state.state_space, None)

    def to_dense(self, on_space):
        if on_space not in ('minimal', 'HilbertSchmidt'):
            raise ValueError("'densitymx_slow' evotype cannot produce Hilbert-space ops!")
        return self.A.toarray()


class OpRepStandard(OpRepDenseSuperop):
    def __init__(self, name, basis, state_space):
        std_unitaries = _itgs.standard_gatename_unitaries()
        self.name = name
        self.basis = basis

        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        superop = _ot.unitary_to_superop(U, basis)
        state_space = _StateSpace.cast(state_space)
        assert(superop.shape[0] == state_space.dim)

        super(OpRepStandard, self).__init__(superop, basis, state_space)


class OpRepKraus(OpRep):
    def __init__(self, basis, kraus_reps, state_space):
        self.basis = basis
        self.kraus_reps = kraus_reps  # superop reps in this evotype (must be reps of *this* evotype)
        state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == state_space.dim)
        super(OpRepKraus, self).__init__(state_space)

    def acton(self, state):
        return _StateRepDense(sum([rep.acton(state).data
                                   for rep in self.kraus_reps]), state.state_space, None)

    def adjoint_acton(self, state):
        return _StateRepDense(sum([rep.adjoint_acton(state).data
                                   for rep in self.kraus_reps]), state.state_space, None)

    def __str__(self):
        return "OpRepKraus with ops\n" + str(self.kraus_reps)

    def copy(self):
        return OpRepKraus(self.basis, list(self.kraus_reps), self.state_space)

    def to_dense(self, on_space):
        assert(on_space in ('minimal', 'HilbertSchmidt')), \
            'Can only compute OpRepKraus.to_dense on HilbertSchmidt space!'
        return sum([rep.to_dense(on_space) for rep in self.kraus_reps])


class OpRepRandomUnitary(OpRep):
    def __init__(self, basis, unitary_rates, unitary_reps, seed_or_state, state_space):
        self.basis = basis
        self.unitary_reps = unitary_reps
        self.unitary_rates = _np.require(unitary_rates.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
        self.state_space = _StateSpace.cast(state_space)
        assert(self.basis.dim == self.state_space.dim)

    def acton(self, state):
        return _StateRepDense(sum([rate * rep.acton(state).data
                                   for rate, rep in zip(self.unitary_rates, self.unitary_reps)]),
                              state.state_space, None)

    def adjoint_acton(self, state):
        return _StateRepDense(sum([rate * rep.adjoint_acton(state).data
                                   for rate, rep in zip(self.unitary_rates, self.unitary_reps)]),
                              state.state_space, None)

    def __str__(self):
        return "OpRepRandomUnitary:\n" + " rates: " + str(self.unitary_rates)  # maybe show ops too?

    def copy(self):
        return OpRepRandomUnitary(self.basis, self.unitary_rates, list(self.unitary_reps), None, self.state_space)

    def update_unitary_rates(self, rates):
        self.unitary_rates[:] = rates

    def to_dense(self, on_space):
        assert(on_space in ('minimal', 'HilbertSchmidt'))  # below code only works in this case
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
        self.basis = basis

        super(OpRepStochastic, self).__init__(basis, _np.array(rates, 'd'), reps, seed_or_state, state_space)

    def update_rates(self, rates):
        unitary_rates = [1 - sum(rates)] + list(rates)
        self.rates[:] = rates
        self.update_unitary_rates(unitary_rates)


class OpRepComposed(OpRep):

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

    def reinit_factor_op_reps(self, factor_reps):
        self.factor_reps = factor_reps


class OpRepSum(OpRep):
    def __init__(self, factor_reps, state_space):
        #assert(len(factor_reps) > 0), "Summed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(OpRepSum, self).__init__(state_space)

    def reinit_factor_reps(self, factor_reps):
        self.factor_reps = factor_reps

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = _StateRepDense(_np.zeros(state.data.shape, 'd'), state.state_space, None)
        for f in self.factor_reps:
            output_state.data += f.acton(state).data
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = _StateRepDense(_np.zeros(state.data.shape, 'd'), state.state_space, None)
        for f in self.factor_reps:
            output_state.data += f.adjoint_acton(state).data
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
        # count possible *density-matrix-space* indices of each component of the tensor product block
        numBasisEls = _np.array([state_space.label_dimension(l) for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        actionInds = _np.array(labelIndices, _np.int64)
        assert(_np.prod([numBasisEls[i] for i in actionInds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        nBlocks = state_space.num_tensor_prod_blocks
        iActiveBlock = iTensorProdBlk
        nComponents = len(state_space.tensor_product_block_labels(iActiveBlock))
        #embeddedDim = embedded_rep.dim
        blocksizes = _np.array([_np.prod(state_space.tensor_product_block_dimensions(k))
                                for k in range(nBlocks)], _np.int64)

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

        #self.embeddedDim = embeddedDim
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
        output_state = _StateRepDense(_np.zeros(state.data.shape, 'd'), state.state_space, None)
        offset = self.offset  # if rel_to_block else self.offset (rel_to_block == False here)

        #print("DB REPLIB ACTON: ",self.basisInds_noop_blankaction)
        #print("DB REPLIB ACTON: ",self.basisInds_action)
        #print("DB REPLIB ACTON: ",self.multipliers)
        for b in _itertools.product(*self.basisInds_noop_blankaction):  # zeros in all action-index locations
            vec_index_noop = _np.dot(self.multipliers, tuple(b))
            inds = []
            for op_b in _itertools.product(*self.basisInds_action):
                vec_index = vec_index_noop
                for i, bInd in zip(self.action_inds, op_b):
                    #b[i] = bInd #don't need to do this; just update vec_index:
                    vec_index += self.multipliers[i] * bInd
                inds.append(offset + vec_index)
            embedded_instate = _StateRepDense(state.data[inds], state.state_space, None)
            embedded_outstate = self.embedded.acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = _StateRepDense(_np.zeros(state.data.shape, 'd'), state.state_space, None)
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
            embedded_instate = _StateRepDense(state.data[inds], state.state_space, None)
            embedded_outstate = self.embedded.adjoint_acton(embedded_instate)
            output_state.data[inds] += embedded_outstate.data

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class OpRepExpErrorgen(OpRep):

    def __init__(self, errorgen_rep):
        state_space = errorgen_rep.state_space
        self.errorgen_rep = errorgen_rep

        #initial values - will be updated by calls to set_exp_params
        self.mu = 1.0
        self.eta = 1.0
        self.m_star = 0
        self.s = 0
        super(OpRepExpErrorgen, self).__init__(state_space)

    def errgenrep_has_changed(self, onenorm_upperbound):
        # don't reset matrix exponential params (based on operator norm) when vector hasn't changed much
        mu, m_star, s, eta = _mt.expop_multiply_prep(
            self.errorgen_rep.aslinearoperator(),
            a_1_norm=onenorm_upperbound)
        self.set_exp_params(mu, eta, m_star, s)

    def set_exp_params(self, mu, eta, m_star, s):
        self.mu = mu
        self.eta = eta
        self.m_star = m_star
        self.s = s

    def exp_params(self):
        return (self.mu, self.eta, self.m_star, self.s)

    def acton(self, state):
        """ Act this gate map on an input state """
        statedata = state.data.copy()  # must COPY because _custom... call below *modifies* "b" arg

        tol = 1e-16  # 2^-53 (=Scipy default) -- TODO: make into an arg?
        A = self.errorgen_rep.aslinearoperator()  # ~= a sparse matrix for call below
        statedata = _mt._custom_expm_multiply_simple_core(
            A, statedata, self.mu, self.m_star, self.s, tol, self.eta)
        return _StateRepDense(statedata, state.state_space, None)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for sparse Lindblad LinearOperator Reps yet.")


class OpRepIdentityPlusErrorgen(OpRep):

    def __init__(self, errorgen_rep):
        state_space = errorgen_rep.state_space
        self.errorgen_rep = errorgen_rep
        super(OpRepIdentityPlusErrorgen, self).__init__(state_space)

    def acton(self, state):
        """ Act this gate map on an input state """
        statedata = state.data + self.errorgen_rep.acton(state).data
        return _StateRepDense(statedata, state.state_space, None)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for OpRepIdentityPlusErrorgen yet.")


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
