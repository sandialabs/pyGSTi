from .. import modelmember as _modelmember

#Note on initialization sequence of Operations within a Model:
# 1) a Model is constructed (empty)
# 2) a LinearOperator is constructed - apart from a Model if it's locally parameterized,
#    otherwise with explicit reference to an existing Model's labels/indices.
#    All gates (ModelMember objs in general) have a "gpindices" member which
#    can either be initialized upon construction or set to None, which signals
#    that the Model must initialize it.
# 3) the LinearOperator is assigned/added to a dict within the Model.  As a part of this
#    process, the LinearOperator's 'gpindices' member is set, if it isn't already, and the
#    Model's "global" parameter vector (and number of params) is updated as
#    needed to accomodate new parameters.
#
# Note: gpindices may be None (before initialization) or any valid index
#  into a 1D numpy array (e.g. a slice or integer array).  It may NOT have
#  any repeated elements.
#
# When a LinearOperator is removed from the Model, parameters only used by it can be
# removed from the Model, and the gpindices members of existing gates
# adjusted as needed.
#
# When derivatives are taken wrt. a model parameter (1 col of a jacobian)
# derivatives wrt each gate that includes that parameter in its gpindices
# must be processed.


class LinearOperator(_modelmember.ModelMember):
    """
    Base class for all operation representations

    Parameters
    ----------
    rep : object
        A representation object containing the core data for this operator.

    evotype : str
        The evolution type of this operator, for matching with forward simulators.

    Attributes
    ----------
    size : int
        Return the number of independent elements in this operation (when viewed as a dense array)

    dirty : bool
        Whether this object has been modified in a way that could have affected its parameters.
        A parent :class:`OpModel` uses this information to know when it needs to refresh it's
        model-wide parameter vector.
    """

    def __init__(self, rep, evotype):
        """ Initialize a new LinearOperator """
        if isinstance(rep, int):  # For operators that have no representation themselves (term ops)
            dim = rep             # allow passing an integer as `rep`.
            rep = None
        else:
            dim = rep.dim
        super(LinearOperator, self).__init__(dim, evotype)
        self._rep = rep

    @property
    def size(self):
        """
        Return the number of independent elements in this operation (when viewed as a dense array)

        Returns
        -------
        int
        """
        return (self.dim)**2

    def set_dense(self, m):
        """
        Set the dense-matrix value of this operation.

        Attempts to modify operation parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the operation action.

        Returns
        -------
        None
        """
        raise ValueError("Cannot set the value of a %s directly!" % self.__class__.__name__)

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.

        For time-independent operators (the default), this function does nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        pass

    #def rep_at_time(self, t):
    #    """
    #    Retrieves a representation of this operator at time `t`.
    #
    #    This is operationally equivalent to calling `self.set_time(t)` and
    #    then retrieving `self._rep`.  However, what is returned from this function
    #    need not be the same rep object for different times, allowing the
    #    operator object to cache many reps for different times to increase performance
    #    (this avoids having to initialize the same rep at a given time).
    #
    #    Parameters
    #    ----------
    #    t : float
    #        The time.
    #
    #    Returns
    #    -------
    #    object
    #    """
    #    self.set_time(t)
    #    return self._rep

    def to_dense(self):
        """
        Return this operation as a dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        raise NotImplementedError("to_dense(...) not implemented for %s objects!" % self.__class__.__name__)

    def acton(self, state):
        """
        Act with this operator upon `state`

        Parameters
        ----------
        state : SPAMVec
            The state to act on

        Returns
        -------
        SPAMVec
            The output state
        """
        from . import spamvec as _sv  # can we move this to top?
        assert(self._rep is not None), "Internal Error: representation is None!"
        assert(state._evotype == self._evotype), "Evolution type mismatch: %s != %s" % (self._evotype, state._evotype)

        #Perform actual 'acton' operation
        output_rep = self._rep.acton(state._rep)

        #Build a SPAMVec around output_rep
        #TODO: make this into a single Static state vec class.
        if self._evotype in ("densitymx", "statevec"):
            return _sv.StaticSPAMVec(output_rep.to_dense(), self._evotype, 'prep')
        else:  # self._evotype == "stabilizer"
            return _sv.StabilizerSPAMVec(sframe=_stabilizer.StabilizerFrame(
                output_rep.smatrix, output_rep.pvectors, output_rep.amps))

    @property
    def dirty(self):
        """
        Whether this operator is "dirty" - i.e. may have had its parameters changed.
        """
        return _modelmember.ModelMember.dirty.fget(self)  # call base class

    #TODO: REMOVE  - we don't need _cachedrep anymore
    #@dirty.setter
    #def dirty(self, value):
    #    """
    #    Whether this operator is "dirty" - i.e. may have had its parameters changed.
    #    """
    #    if value:
    #        self._cachedrep = None  # clear cached rep
    #    _modelmember.ModelMember.dirty.fset(self, value)  # call base class setter
    #def __getstate__(self):
    #    st = super(LinearOperator, self).__getstate__()
    #    st['_cachedrep'] = None  # can't pickle this!
    #    return st
    #def copy(self, parent=None, memo=None):
    #    """
    #    Copy this LinearOperator.
    #
    #    Parameters
    #    ----------
    #    parent : Model, optional
    #        The parent model to set for the copy.
    #    """
    #    self._cachedrep = None  # deepcopy in ModelMember.copy can't copy CReps!
    #    return _modelmember.ModelMember.copy(self, parent, memo)

    def to_sparse(self):
        """
        Return this operation as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        raise NotImplementedError("to_sparse(...) not implemented for %s objects!" % self.__class__.__name__)

    def frobeniusdist_squared(self, other_op, transform=None, inv_transform=None):
        """
        Return the squared frobenius difference between this operation and `other_op`

        Optionally transforms this operation first using matrices
        `transform` and `inv_transform`.  Specifically, this operation gets
        transfomed as: `O => inv_transform * O * transform` before comparison with
        `other_op`.

        Parameters
        ----------
        other_op : DenseOperator
            The other operation.

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `transform`.

        Returns
        -------
        float
        """
        if transform is None and inv_transform is None:
            return _gt.frobeniusdist_squared(self.to_dense(), other_op.to_dense())
        else:
            return _gt.frobeniusdist_squared(_np.dot(
                inv_transform, _np.dot(self.to_dense(), transform)),
                other_op.to_dense())

    def frobeniusdist(self, other_op, transform=None, inv_transform=None):
        """
        Return the frobenius distance between this operation and `other_op`.

        Optionally transforms this operation first using matrices
        `transform` and `inv_transform`.  Specifically, this operation gets
        transfomed as: `O => inv_transform * O * transform` before comparison with
        `other_op`.

        Parameters
        ----------
        other_op : DenseOperator
            The other operation.

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `transform`.

        Returns
        -------
        float
        """
        return _np.sqrt(self.frobeniusdist_squared(other_op, transform, inv_transform))

    def residuals(self, other_op, transform=None, inv_transform=None):
        """
        The per-element difference between this `DenseOperator` and `other_op`.

        Optionally, tansforming this operation first as
        `O => inv_transform * O * transform`.

        Parameters
        ----------
        other_op : DenseOperator
            The operation to compare against.

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `transform`.

        Returns
        -------
        numpy.ndarray
            A 1D-array of size equal to that of the flattened operation matrix.
        """
        if transform is None and inv_transform is None:
            return _gt.residuals(self.to_dense(), other_op.to_dense())
        else:
            return _gt.residuals(_np.dot(
                inv_transform, _np.dot(self.to_dense(), transform)),
                other_op.to_dense())

    def jtracedist(self, other_op, transform=None, inv_transform=None):
        """
        Return the Jamiolkowski trace distance between this operation and `other_op`.

        Optionally, tansforming this operation first as
        `O => inv_transform * O * transform`.

        Parameters
        ----------
        other_op : DenseOperator
            The operation to compare against.

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `transform`.

        Returns
        -------
        float
        """
        if transform is None and inv_transform is None:
            return _gt.jtracedist(self.to_dense(), other_op.to_dense())
        else:
            return _gt.jtracedist(_np.dot(
                inv_transform, _np.dot(self.to_dense(), transform)),
                other_op.to_dense())

    def diamonddist(self, other_op, transform=None, inv_transform=None):
        """
        Return the diamond distance between this operation and `other_op`.

        Optionally, tansforming this operation first as
        `O => inv_transform * O * transform`.

        Parameters
        ----------
        other_op : DenseOperator
            The operation to compare against.

        transform : numpy.ndarray, optional
            Transformation matrix.

        inv_transform : numpy.ndarray, optional
            Inverse of `transform`.

        Returns
        -------
        float
        """
        if transform is None and inv_transform is None:
            return _gt.diamonddist(self.to_dense(), other_op.to_dense())
        else:
            return _gt.diamonddist(_np.dot(
                inv_transform, _np.dot(self.to_dense(), transform)),
                other_op.to_dense())

    def transform_inplace(self, s):
        """
        Update operation matrix `O` with `inv(s) * O * s`.

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        In this particular case *any* transform of the appropriate
        dimension is possible, since all operation matrix elements are parameters.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        Smx = s.transform_matrix
        Si = s.transform_matrix_inverse
        self.set_dense(_np.dot(Si, _np.dot(self.to_dense(), Smx)))

    def depolarize(self, amount):
        """
        Depolarize this operation by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the operation such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the operation parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the operation. In standard
            bases, depolarization corresponds to multiplying the operation matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        if isinstance(amount, float):
            D = _np.diag([1] + [1 - amount] * (self.dim - 1))
        else:
            assert(len(amount) == self.dim - 1)
            D = _np.diag([1] + list(1.0 - _np.array(amount, 'd')))
        self.set_dense(_np.dot(D, self.to_dense()))

    def rotate(self, amount, mx_basis="gm"):
        """
        Rotate this operation by the given `amount`.

        Generally, the rotate function updates the *parameters* of
        the operation such that the resulting operation matrix is rotated.  If
        such an update cannot be done (because the operation parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The operation's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        rotnMx = _gt.rotation_gate_mx(amount, mx_basis)
        self.set_dense(_np.dot(rotnMx, self.to_dense()))

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
        if(self.num_params != 0):
            raise NotImplementedError("Default deriv_wrt_params is only for 0-parameter (default) case (%s)"
                                      % str(self.__class__.__name__))

        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.size, 0), dtype)
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.num_params > 0

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this operation with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1 : list or numpy.ndarray
            List of parameter indices to take 1st derivatives with respect to.
            (None means to use all the this operation's parameters.)

        wrt_filter2 : list or numpy.ndarray
            List of parameter indices to take 2nd derivatives with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        if not self.has_nonzero_hessian():
            return _np.zeros((self.size, self.num_params, self.num_params), 'd')

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)

    ##Pickle plumbing

    def __setstate__(self, state):
        self.__dict__.update(state)

    #Note: no __str__ fn

    @staticmethod
    def convert_to_matrix(m):
        """
        Static method that converts a matrix-like object to a 2D numpy array.

        Parameters
        ----------
        m : array_like
            matrix-like object

        Returns
        -------
        numpy array
        """
        if isinstance(m, LinearOperator):
            dim = m.dim
            matrix = _np.asarray(m).copy()
            # LinearOperator objs should also derive from ndarray
        elif isinstance(m, _np.ndarray):
            matrix = m.copy()
        else:
            try:
                dim = len(m)
                len(m[0])
                # XXX this is an abuse of exception handling
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % m)
            if any([len(row) != dim for row in m]):
                raise ValueError("%s is not a *square* 2D array" % m)

            ar = _np.array(m)
            if _np.all(_np.isreal(ar)):
                matrix = _np.array(ar.real, 'd')
            else:
                matrix = _np.array(ar, 'complex')

        if len(matrix.shape) != 2:
            raise ValueError("%s has %d dimensions when 2 are expected"
                             % (m, len(matrix.shape)))

        if matrix.shape[0] != matrix.shape[1]:  # checked above, but just to be safe
            raise ValueError("%s is not a *square* 2D array" % m)  # pragma: no cover

        return matrix

    def get_chp_str(self, targets=None):
        """Return a string suitable for printing to a CHP input file after
        probabilistically selecting operation.

        Parameters
        ----------
        targets: list of int, optional
            Qubits to be applied to (if None, uses stored CHP strings directly)

        Returns
        -------
        s : str
            String of CHP code
        """
        assert (self._evotype == 'chp'), 'Only "chp" evotype can use get_chp_str'

        ops = self._rep.chp_ops
        nqubits = self._rep.nqubits

        if targets is not None:
            assert len(targets) == nqubits, "Got {0} targets instead of required {1}".format(len(targets), nqubits)
            target_map = {str(i): str(t) for i, t in enumerate(targets)}

        s = ""
        for op in ops:
            # Substitute if alternate targets provided
            if targets is not None:
                op_str = ''.join([target_map[c] if c in target_map else c for c in op])
            else:
                op_str = op

            s += op_str + '\n'

        return s
