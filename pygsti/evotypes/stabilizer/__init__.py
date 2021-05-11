


class OpRepComposed(XXX):


    def __init__(self, factor_op_reps, dim):
        nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
        #rep = replib.SBOpRepComposed(factor_op_reps, nQubits)

    def reinit_factor_op_reps(factor_op_reps):
        pass  # TODO


class OpRepEmbedded(XXX):

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
        rep = replib.SBOpRepEmbedded(embedded_rep, nQubits, qubit_indices)


class OpRepSum(XXX):
    def __init__(self, factor_reps, dim):
        pass

    def reinit_factor_reps(self, factor_reps):
        pass


class OpRepClifford(XXX):
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

        #Update members so they reference the same (contiguous) memory as the rep
        
        self._dense_unitary = _np.ascontiguousarray(unitarymx, complex)
        self.smatrix = _np.ascontiguousarray(self.smatrix, _np.int64)
        self.svector = _np.ascontiguousarray(self.svector, _np.int64)
        self.inv_smatrix = _np.ascontiguousarray(self.inv_smatrix, _np.int64)
        self.inv_svector = _np.ascontiguousarray(self.inv_svector, _np.int64)

        #Create representation
        rep = replib.SBOpRepClifford(self.smatrix, self.svector,
                                     self.inv_smatrix, self.inv_svector,
                                     self._dense_unitary)

    def __str__(self):
        """ Return string representation """
        s = "Clifford operation with matrix:\n"
        s += _mt.mx_to_string(self.smatrix, width=2, prec=0)
        s += " and vector " + _mt.mx_to_string(self.svector, width=2, prec=0)
        return s


class OpRepRepeated(XXX):
    def __init__(self, rep_to_repeat, num_repetitions, dim):
        nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
        rep = replib.SVOpRepExponentiated(self.exponentiated_op._rep, self.power, nQubits)

