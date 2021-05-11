


class OpRepDense(_evotype.OpRepDense):
    pass  # like replib.DMOpRepDense(mx)
    # must have a .base property that is a numpy array - this is how updates are performed on this type
    
    def __init__(self, dim):
        mx = _np.require(_np.identity(dim, 'd'),
                         requirements=['OWNDATA', 'C_CONTIGUOUS'])
        # in statevec, type should be complex

        
                 
        dtype = complex if evotype == "statevec" else 'd'
        mx = _np.ascontiguousarray(mx, dtype)  # may not give mx it's own data
        mx = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])

        if evotype == "statevec":
            rep = replib.SVOpRepDense(mx)
        elif evotype == "densitymx":
            rep = 
        else:
            raise ValueError("Invalid evotype for a DenseOperator: %s" % evotype)

#OR for cython, in a pyx file:
cdef class OpRepDense(_evotype.OpRepDense_cython):
    pass

# for reference, from fastreplib.pyx
cdef class DMOpRepDense(DMOpRep):
    cdef public np.ndarray base

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] data, int reducefix=0):
        if reducefix == 0:
            self.base = data  # the usual case - just take data ptr
        else:
            # because serialization of numpy array flags is borked (around Numpy v1.16), we need to copy data
            # (so self.base *owns* it's data) and manually convey the writeable flag.
            self.base = np.require(data.copy(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
            self.base.flags.writeable = True if reducefix == 1 else False

        #print("PYX dense gate constructed w/dim ",data.shape[0])
        self.c_op = new DMOpCRep_Dense(<double*>self.base.data,
                                           <INT>self.base.shape[0])

    def __reduce__(self):
        reducefix = 1 if self.base.flags.writeable else 2
        return (DMOpRepDense, (self.base, reducefix))

    def __str__(self):
        s = ""
        cdef DMOpCRep_Dense* my_cgate = <DMOpCRep_Dense*>self.c_op # b/c we know it's a _Dense gate...
        cdef INT i,j,k
        for i in range(my_cgate._dim):
            k = i*my_cgate._dim
            for j in range(my_cgate._dim):
                s += str(my_cgate._dataptr[k+j]) + " "
            s += "\n"
        return s

    def copy(self):
        return DMOpRepDense(self.base.copy())


class OpRepComposed(XXX):

    def __init__(self, factor_op_reps, dim):
        pass

    def reinit_factor_op_reps(factor_op_reps):
        pass  # TODO


class OpRepEmbedded(XXX):

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

        nBlocks = state_space_labels.num_tensor_prod_blocks()
        iActiveBlock = iTensorProdBlk
        nComponents = len(state_space_labels.labels[iActiveBlock])
        embeddedDim = embedded_rep.dim
        blocksizes = _np.array([_np.product(state_space_labels.tensor_product_block_dims(k))
                                for k in range(nBlocks)], _np.int64)
        rep = replib.DMOpRepEmbedded(embedded_rep,
                                     numBasisEls, actionInds, blocksizes, embeddedDim,
                                     nComponents, iActiveBlock, nBlocks, opDim)


class OpRepExpErrorgen(XXX):

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
        
        mu, m_star, s, eta = 1.0, 0, 0, 1.0  # initial values - will be updated by call to _update_rep below
        rep = replib.DMOpRepLindblad(errorgen_rep,
                                     mu, eta, m_star, s,
                                     Udata, Uindices, Uindptr)

    def errgenrep_has_changed(self):
        # don't reset matrix exponential params (based on operator norm) when vector hasn't changed much
        mu, m_star, s, eta = _mt.expop_multiply_prep(
            self.errorgen_rep.aslinearoperator(),
            a_1_norm=self.errorgen_rep.onenorm_upperbound())  # need to add ths to rep class from op TODO!!!
        self.set_exp_params(mu, eta, m_star, s)


class OpRepStochastic(XXX):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state):
        self.basis = basis
        self.stochastic_superops = []
        for b in self.basis.elements[1:]:
            std_superop = _lbt.nonham_lindbladian(b, b, sparse=False)
            self.stochastic_superops.append(_bt.change_basis(std_superop, 'std', self.basis))

        #init DENSE rep for now
        rep = replib.DMOpRepDense(_np.ascontiguousarray(_np.identity(dim, 'd')))
        # setup base
        self.update_rates(initial_rates)

    def update_rates(self, rates):
        errormap = _np.identity(self.basis.dim)
        for rate, ss in zip(rates, self.stochastic_superops):
            errormap += rate * ss
        self.base[:, :] = errormap

    def to_dense(self):  # TODO - put this in all reps?  - used in stochastic op...
        # DEFAULT: raise NotImplementedError('No to_dense implemented for evotype "%s"' % self._evotype)
        return self._rep.base  # copy?


class OpRepSum(XXX):
    # similar to DMOpRepSum
    def __init__(self, factor_reps, dim):
        pass

    def reinit_factor_reps(self, factor_reps):
        pass

class OpRepRepeated(XXX):
    def __init__(self, rep_to_repeat, num_repetitions, dim):
        #similar to:
        rep = replib.DMOpRepExponentiated(self.exponentiated_op._rep, self.power, dim)


class OpRepStandard(XXX):
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
        rep = replib.DMOpRepDense(LinearOperator.convert_to_matrix(ptm))


class OpRepSparse(XXX):
    def __init__(data, index, indptr):
        pass  # see replib.DMOpRepSparse
