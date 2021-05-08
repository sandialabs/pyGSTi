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
