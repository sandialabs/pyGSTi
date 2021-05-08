class CliffordOp(LinearOperator):
    """
    A Clifford operation, represented via a symplectic matrix.

    Parameters
    ----------
    unitary : numpy.ndarray
        The unitary action of the clifford operation.

    symplecticrep : tuple, optional
        A (symplectic matrix, phase vector) 2-tuple specifying the pre-
        computed symplectic representation of `unitary`.  If None, then
        this representation is computed automatically from `unitary`.
    """

    def __init__(self, unitary, symplecticrep=None):
        """
        Creates a new CliffordOp from a unitary operation.

        Note: while the clifford operation is held internally in a symplectic
        representation, it is also be stored as a unitary (so the `unitary`
        argument is required) for keeping track of global phases when updating
        stabilizer frames.

        If a non-Clifford unitary is specified, then a ValueError is raised.

        Parameters
        ----------
        unitary : numpy.ndarray
            The unitary action of the clifford operation.

        symplecticrep : tuple, optional
            A (symplectic matrix, phase vector) 2-tuple specifying the pre-
            computed symplectic representation of `unitary`.  If None, then
            this representation is computed automatically from `unitary`.

        """
        #self.superop = superop
        self.unitary = unitary
        assert(self.unitary is not None), "Must supply `unitary` argument!"

        #if self.superop is not None:
        #    assert(unitary is None and symplecticrep is None),"Only supply one argument to __init__"
        #    raise NotImplementedError("Superop -> Unitary calc not implemented yet")

        if symplecticrep is not None:
            self.smatrix, self.svector = symplecticrep
        else:
            # compute symplectic rep from unitary
            self.smatrix, self.svector = _symp.unitary_to_symplectic(self.unitary, flagnonclifford=True)

        self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
            self.smatrix, self.svector)  # cache inverse since it's expensive

        #nQubits = len(self.svector) // 2
        #dim = 2**nQubits  # "stabilizer" is a "unitary evolution"-type mode

        #Update members so they reference the same (contiguous) memory as the rep
        U = self.unitary.to_dense() if isinstance(self.unitary, LinearOperator) else self.unitary
        self._dense_unitary = _np.ascontiguousarray(U, complex)
        self.smatrix = _np.ascontiguousarray(self.smatrix, _np.int64)
        self.svector = _np.ascontiguousarray(self.svector, _np.int64)
        self.inv_smatrix = _np.ascontiguousarray(self.inv_smatrix, _np.int64)
        self.inv_svector = _np.ascontiguousarray(self.inv_svector, _np.int64)

        #Create representation
        rep = replib.SBOpRepClifford(self.smatrix, self.svector,
                                     self.inv_smatrix, self.inv_svector,
                                     self._dense_unitary)
        LinearOperator.__init__(self, rep, "stabilizer")

    #NOTE: if this operation had parameters, we'd need to clear inv_smatrix & inv_svector
    # whenever the smatrix or svector changed, respectively (probably in from_vector?)

    #def torep(self):
    #    """
    #    Return a "representation" object for this operation.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self.inv_smatrix is None or self.inv_svector is None:
    #        self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
    #            self.smatrix, self.svector)  # cache inverse since it's expensive
    #
    #    invs, invp = self.inv_smatrix, self.inv_svector
    #    U = self.unitary.todense() if isinstance(self.unitary, LinearOperator) else self.unitary
    #    return replib.SBOpRepClifford(_np.ascontiguousarray(self.smatrix, _np.int64),
    #                                   _np.ascontiguousarray(self.svector, _np.int64),
    #                                   _np.ascontiguousarray(invs, _np.int64),
    #                                   _np.ascontiguousarray(invp, _np.int64),
    #                                   _np.ascontiguousarray(U, complex))

    def __str__(self):
        """ Return string representation """
        s = "Clifford operation with matrix:\n"
        s += _mt.mx_to_string(self.smatrix, width=2, prec=0)
        s += " and vector " + _mt.mx_to_string(self.svector, width=2, prec=0)
        return s
