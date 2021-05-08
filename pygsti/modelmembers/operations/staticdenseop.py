class StaticDenseOp(DenseOperator):
    """
    An operation matrix that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    m : array_like or LinearOperator
        a square 2D array-like or LinearOperator object representing the operation action.
        The shape of m sets the dimension of the operation.

    evotype : {"statevec", "densitymx", "auto"}
        The evolution type.  If `"auto"`, then `"statevec"` is used if and only if `m`
        has a complex datatype.
    """

    def __init__(self, m, evotype="auto"):
        """
        Initialize a StaticDenseOp object.

        Parameters
        ----------
        m : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the operation action.
            The shape of m sets the dimension of the operation.

        evotype : {"statevec", "densitymx", "auto"}
            The evolution type.  If `"auto"`, then `"statevec"` is used if and only if `m`
            has a complex datatype.
        """
        m = LinearOperator.convert_to_matrix(m)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(m) else "densitymx"
        assert(evotype in ("statevec", "densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)
        DenseOperator.__init__(self, m, evotype)
        #(default DenseOperator/LinearOperator methods implement an object with no parameters)

        #if self._evotype == "svterm": # then we need to extract unitary
        #    op_std = _bt.change_basis(operation, basis, 'std')
        #    U = _gt.process_mx_to_unitary(self)

    def compose(self, other_op):
        """
        Compose this operation with another :class:`StaticDenseOp`.

        Create and return a new operation that is the composition of this operation
        followed by other_op, which *must be another StaticDenseOp*.
        (For more general compositions between different types of operations, use
        the module-level compose function.)  The returned operation's matrix is
        equal to dot(this, other_op).

        Parameters
        ----------
        other_op : StaticDenseOp
            The operation to compose to the right of this one.

        Returns
        -------
        StaticDenseOp
        """
        return StaticDenseOp(_np.dot(self.base, other_op.base), self._evotype)
