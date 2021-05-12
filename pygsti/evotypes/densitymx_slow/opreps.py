"""
Operation representation classes for the `densitymx` evolution type.
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


class OpRep(_basereps.OpRep):
    def __init__(self, dim):
        self.dim = dim

    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()

    def aslinearoperator(self):
        def mv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = StateRep(_np.ascontiguousarray(v, 'd'))
            return self.acton(in_state).to_dense()

        def rmv(v):
            if v.ndim == 2 and v.shape[1] == 1: v = v[:, 0]
            in_state = StateRep(_np.ascontiguousarray(v, 'd'))
            return self.adjoint_acton(in_state).to_dense()
        return LinearOperator((self.dim, self.dim), matvec=mv, rmatvec=rmv)  # transpose, adjoint, dot, matmat?


class OpRepDense(OpRep):
    def __init__(self, dim):
        self.base = _np.require(_np.identity(dim, 'd'),
                                requirements=['OWNDATA', 'C_CONTIGUOUS'])
        super(OpRepDense, self).__init__(self.base.shape[0])

    def acton(self, state):
        return StateRep(_np.dot(self.base, state.base))

    def adjoint_acton(self, state):
        return StateRep(_np.dot(self.base.T, state.base))  # no conjugate b/c *real* data

    def __str__(self):
        return "OpRepDense:\n" + str(self.base)


class OpRepSparse(OpRep):
    def __init__(self, a_data, a_indices, a_indptr):
        dim = len(a_indptr) - 1
        self.A = _sps.csr_matrix((a_data, a_indices, a_indptr), shape=(dim, dim))
        super(OpRepSparse, self).__init__(dim)

    def __reduce__(self):
        return (OpRepSparse, (self.A.data, self.A.indices, self.A.indptr))

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
        return StateRep(self.A.dot(state.base))

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        Aadj = self.A.conjugate(copy=True).transpose()
        return StateRep(Aadj.dot(state.base))


class OpRepStandard(OpRepDense):
    def __init__(self, name):
        std_unitaries = _itgs.standard_gatename_unitaries()
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]

        #TODO: for statevec:
        #if evotype == 'statevec':
        #    rep = replib.SVOpRepDense(LinearOperator.convert_to_matrix(U))
        #else:  # evotype in ('densitymx', 'svterm', 'cterm')

        ptm = _gt.unitary_to_pauligate(U)
        super(OpRepStandard, self).__init__(ptm.shape[0])
        self.base[:, :] = LinearOperator.convert_to_matrix(ptm)


class OpRepStochastic(OpRepDense):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state):
        self.basis = basis
        self.stochastic_superops = []
        for b in self.basis.elements[1:]:
            std_superop = _lbt.nonham_lindbladian(b, b, sparse=False)
            self.stochastic_superops.append(_bt.change_basis(std_superop, 'std', self.basis))

        super(OpRepStochastic, self).__init__(self.basis.dim)
        self.update_rates(initial_rates)

    def update_rates(self, rates):
        errormap = _np.identity(self.basis.dim)
        for rate, ss in zip(rates, self.stochastic_superops):
            errormap += rate * ss
        self.base[:, :] = errormap

    def to_dense(self):  # TODO - put this in all reps?  - used in stochastic op...
        # DEFAULT: raise NotImplementedError('No to_dense implemented for evotype "%s"' % self._evotype)
        return self._rep.base  # copy?


#class OpRepClifford(OpRep):  # TODO?
#    #def __init__(self, unitarymx, symplecticrep):
#    #    pass


class OpRepComposed(OpRep):

    def __init__(self, factor_op_reps, dim):
        #assert(len(factor_op_reps) > 0), "Composed gates must contain at least one factor gate!"
        self.factor_reps = factor_op_reps
        super(DMOpRepComposed, self).__init__(dim)

    def __reduce__(self):
        return (OpRepComposed, (self.factor_reps, self.dim))

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
    def __init__(self, factor_reps, dim):
        #assert(len(factor_reps) > 0), "Summed gates must contain at least one factor gate!"
        self.factor_reps = factor_reps
        super(OpRepSum, self).__init__(dim)

    def reinit_factor_reps(self, factor_reps):
        self.factor_reps = factor_reps

    def __reduce__(self):
        return (DMOpRepSum, (self.factor_reps, self.dim))

    def acton(self, state):
        """ Act this gate map on an input state """
        output_state = StateRep(_np.zeros(state.base.shape, 'd'))
        for f in self.factor_reps:
            output_state.base += f.acton(state).base
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        output_state = StateRep(_np.zeros(state.base.shape, 'd'))
        for f in self.factor_reps:
            output_state.base += f.adjoint_acton(state).base
        return output_state


class OpRepEmbedded(OpRep):

    def __init__(self, state_space_labels, target_labels, embedded_rep):

        iTensorProdBlks = [state_space_labels.tpb_index[label] for label in target_labels]
        # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) != 1:
            raise ValueError("All qubit labels of a multi-qubit operation must correspond to the"
                             " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

        iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
        tensorProdBlkLabels = state_space_labels.labels[iTensorProdBlk]
        # count possible *density-matrix-space* indices of each component of the tensor product block
        numBasisEls = _np.array([state_space_labels.labeldims[l] for l in tensorProdBlkLabels], _np.int64)

        # Separate the components of the tensor product that are not operated on, i.e. that our
        # final map just acts as identity w.r.t.
        labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
        actionInds = _np.array(labelIndices, _np.int64)
        assert(_np.product([numBasisEls[i] for i in actionInds]) == embedded_rep.dim), \
            "Embedded operation has dimension (%d) inconsistent with the given target labels (%s)" % (
                embedded_rep.dim, str(target_labels))

        dim = state_space_labels.dim
        nBlocks = state_space_labels.num_tensor_prod_blocks()
        iActiveBlock = iTensorProdBlk
        nComponents = len(state_space_labels.labels[iActiveBlock])
        embeddedDim = embedded_rep.dim
        blocksizes = _np.array([_np.product(state_space_labels.tensor_product_block_dims(k))
                                for k in range(nBlocks)], _np.int64)

        self.embedded_rep = embedded_rep
        self.num_basis_els = numBasisels
        self.action_inds = actionInds
        self.blocksizes = blocksizes

        num_basis_els_noop_blankaction = num_basis_els.copy()
        for i in action_inds: num_basis_els_noop_blankaction[i] = 1
        self.basisInds_noop_blankaction = [list(range(n)) for n in num_basis_els_noop_blankaction]

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        self.multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(num_basis_els[1:]))))), _np.int64)
        self.basisInds_action = [list(range(num_basis_els[i])) for i in action_inds]

        self.embeddedDim = embeddedDim
        self.ncomponents = nComponents  # number of components in "active" block
        self.active_block_index = iActiveBlock
        self.nblocks = nBlocks
        self.offset = sum(blocksizes[0:active_block_index])
        super(OpRepEmbedded, self).__init__(dim)

    #def __reduce__(self):
    #    return (DMOpRepEmbedded, (self.embedded,
    #                              self.num_basis_els, self.action_inds,
    #                              self.blocksizes, self.embeddedDim,
    #                              self.ncomponents, self.active_block_index,
    #                              self.nblocks, self.dim))

    def _acton_other_blocks_trivially(self, output_state, state):
        offset = 0
        for iBlk, blockSize in enumerate(self.blocksizes):
            if iBlk != self.active_block_index:
                output_state.base[offset:offset + blockSize] = state.base[offset:offset + blockSize]  # identity op
            offset += blockSize

    def acton(self, state):
        output_state = StateRep(_np.zeros(state.base.shape, 'd'))
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
            embedded_instate = DMStateRep(state.base[inds])
            embedded_outstate = self.embedded.acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state

    def adjoint_acton(self, state):
        """ Act the adjoint of this gate map on an input state """
        #NOTE: Same as acton except uses 'adjoint_acton(...)' below
        output_state = DMStateRep(_np.zeros(state.base.shape, 'd'))
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
            embedded_instate = DMStateRep(state.base[inds])
            embedded_outstate = self.embedded.adjoint_acton(embedded_instate)
            output_state.base[inds] += embedded_outstate.base

        #act on other blocks trivially:
        self._acton_other_blocks_trivially(output_state, state)
        return output_state


class OpRepExpErrorgen(OpRep):

    def __init__(self, errorgen_rep):
        
        #self.unitary_postfactor = unitary_postfactor  # can be None
        #self.err_gen_prep = None REMOVE

        #Pre-compute the exponential of the error generator if dense matrices
        # are used, otherwise cache prepwork for sparse expm calls

        #Allocate sparse matrix arrays for rep
        #if self.unitary_postfactor is None:
        #    Udata = _np.empty(0, 'd')
        #    Uindices = _np.empty(0, _np.int64)
        #    Uindptr = _np.zeros(1, _np.int64)
        #else:
        #    assert(_sps.isspmatrix_csr(self.unitary_postfactor)), \
        #        "Internal error! Unitary postfactor should be a *sparse* CSR matrix!"
        #    Udata = self.unitary_postfactor.data
        #    Uindptr = _np.ascontiguousarray(self.unitary_postfactor.indptr, _np.int64)
        #    Uindices = _np.ascontiguousarray(self.unitary_postfactor.indices, _np.int64)

        #TODO REMOVE
        #if len(unitarypost_data) > 0:  # (nnz > 0)
        #    self.unitary_postfactor = _sps.csr_matrix(
        #        (unitarypost_data, unitarypost_indices,
        #         unitarypost_indptr), shape=(dim, dim))
        #else:
        #    self.unitary_postfactor = None  # no unitary postfactor

        dim = errorgen_rep.dim
        self.errorgen_rep = errorgen_rep
        
        #initial values - will be updated by calls to set_exp_params
        self.mu = 1.0
        self.eta = 1.0
        self.m_star = 0
        self.s = 0
        super(OpRepExpErrorgen, self).__init__(dim)

    def errgenrep_has_changed(self):
        # don't reset matrix exponential params (based on operator norm) when vector hasn't changed much
        mu, m_star, s, eta = _mt.expop_multiply_prep(
            self.errorgen_rep.aslinearoperator(),
            a_1_norm=self.errorgen_rep.onenorm_upperbound())  # need to add ths to rep class from op TODO!!!
        self.set_exp_params(mu, eta, m_star, s)

    def set_exp_params(self, mu, eta, m_star, s):
        self.mu = mu
        self.eta = eta
        self.m_star = m_star
        self.s = s

    def get_exp_params(self):
        return (self.mu, self.eta, self.m_star, self.s)

    #def __reduce__(self):
    #    if self.unitary_postfactor is None:
    #        return (DMOpRepLindblad, (self.errorgen_rep, self.mu, self.eta, self.m_star, self.s,
    #                                  _np.empty(0, 'd'), _np.empty(0, _np.int64), _np.zeros(1, _np.int64)))
    #    else:
    #        return (DMOpRepLindblad, (self.errorgen_rep, self.mu, self.eta, self.m_star, self.s,
    #                                  self.unitary_postfactor.data, self.unitary_postfactor.indices,
    #                                  self.unitary_postfactor.indptr))

    def acton(self, state):
        """ Act this gate map on an input state """
        #TODO REMOVE
        #if self.unitary_postfactor is not None:
        #    statedata = self.unitary_postfactor.dot(state.base)
        #else:
        statedata = state.base

        tol = 1e-16  # 2^-53 (=Scipy default) -- TODO: make into an arg?
        A = self.errorgen_rep.aslinearoperator()  # ~= a sparse matrix for call below
        statedata = _mt._custom_expm_multiply_simple_core(
            A, statedata, self.mu, self.m_star, self.s, tol, self.eta)
        return StateRep(statedata)

    def adjoint_acton(self, state):
        """ Act the adjoint of this operation matrix on an input state """
        raise NotImplementedError("No adjoint action implemented for sparse Lindblad LinearOperator Reps yet.")


class OpRepRepeated(OpRep):
    def __init__(self, rep_to_repeat, num_repetitions, dim):
        self.repeated_rep = rep_to_repeat
        self.num_repetitions = num_repetitions
        super(OpRepRepeated, self).__init__(dim)

    def __reduce__(self):
        return (OpRepRepeated, (self.repeated_rep, self.num_repetitions, self.dim))

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
