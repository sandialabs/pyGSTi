
class DenseOperatorInterface(object):
    """
    Adds a numpy-array-mimicing interface onto an operation object.

    The object's ._rep must be a *dense* representation (with a
    .base that is a numpy array).

    This class is distinct from DenseOperator because there are some
    operators, e.g. LindbladOp, that *can* but don't *always* have
    a dense representation.  With such types, a base class allows
    a 'dense_rep' argument to its constructor and a derived class
    sets this to True *and* inherits from DenseOperatorInterface.
    If would not be appropriate to inherit from DenseOperator because
    this is a standalone operator with it's own (dense) ._rep, etc.
    """

    def __init__(self):
        pass

    @property
    def _ptr(self):
        return self._rep.base

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Constructs a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        return finite_difference_deriv_wrt_params(self, wrt_filter, eps=1e-7)

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Note: for efficiency, this doesn't copy the underlying data, so
        the caller should copy this data before modifying it.

        Returns
        -------
        numpy.ndarray
        """
        return _np.asarray(self._ptr)
        # *must* be a numpy array for Cython arg conversion

    def to_sparse(self):
        """
        Return the operation as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return _sps.csr_matrix(self.to_dense())

    def __copy__(self):
        # We need to implement __copy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __copy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        # We need to implement __deepcopy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __deepcopy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy

    #Access to underlying ndarray
    def __getitem__(self, key):
        self.dirty = True
        return self._ptr.__getitem__(key)

    def __getslice__(self, i, j):
        self.dirty = True
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        return self._ptr.__setitem__(key, val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        ret = getattr(self.__dict__['_rep'].base, attr)
        self.dirty = True
        return ret

    #Mimic array behavior
    def __pos__(self): return self._ptr
    def __neg__(self): return -self._ptr
    def __abs__(self): return abs(self._ptr)
    def __add__(self, x): return self._ptr + x
    def __radd__(self, x): return x + self._ptr
    def __sub__(self, x): return self._ptr - x
    def __rsub__(self, x): return x - self._ptr
    def __mul__(self, x): return self._ptr * x
    def __rmul__(self, x): return x * self._ptr
    def __truediv__(self, x): return self._ptr / x
    def __rtruediv__(self, x): return x / self._ptr
    def __floordiv__(self, x): return self._ptr // x
    def __rfloordiv__(self, x): return x // self._ptr
    def __pow__(self, x): return self._ptr ** x
    def __eq__(self, x): return self._ptr == x
    def __len__(self): return len(self._ptr)
    def __int__(self): return int(self._ptr)
    def __long__(self): return int(self._ptr)
    def __float__(self): return float(self._ptr)
    def __complex__(self): return complex(self._ptr)


class BasedDenseOperatorInterface(DenseOperatorInterface):
    """
    A DenseOperatorInterface that uses self.base instead of self._rep.base as the "base pointer" to data.

    This is used by the TPDenseOp class, for example, which has a .base
    that is different from its ._rep.base.
    """
    @property
    def _ptr(self):
        return self.base


class DenseOperator(BasedDenseOperatorInterface, LinearOperator):
    """
    An operator that behaves like a dense operation matrix.

    This class is the common base class for more specific dense operators.

    Parameters
    ----------
    mx : numpy.ndarray
        The operation as a dense process matrix.

    evotype : {"statevec", "densitymx"}
        The evolution type.

    Attributes
    ----------
    base : numpy.ndarray
        Direct access to the underlying process matrix data.
    """

    def __init__(self, mx, evotype):
        """ Initialize a new LinearOperator """
        evotype = _Evotype.cast(evotype)
        rep = evotype.create_dense_rep(mx.shape[0])
        rep.base[:, :] = mx
        LinearOperator.__init__(self, rep, evotype)
        BasedDenseOperatorInterface.__init__(self)
        # "Based" interface requires this and derived classes to have a .base attribute
        # or property that points to the data to interface with.  This gives derived classes
        # flexibility in defining something other than self._rep.base to be used (see TPDenseOp).

    @property
    def base(self):
        """
        The underlying dense process matrix.
        """
        return self._rep.base

    def __str__(self):
        s = "%s with shape %s\n" % (self.__class__.__name__, str(self.base.shape))
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s
