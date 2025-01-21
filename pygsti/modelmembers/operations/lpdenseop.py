"""
The LinearlyParamArbitraryOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.modelmembers.operations.denseop import DenseOperator as _DenseOperator
from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.tools import matrixtools as _mt

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


class LinearlyParameterizedElementTerm(object):
    """
    Encapsulates a single term within a LinearlyParamArbitraryOp.

    Parameters
    ----------
    coeff : float, optional
        The term's coefficient

    param_indices : list
        A list of integers, specifying which parameters are muliplied
        together (and finally, with `coeff`) to form this term.
    """

    def __init__(self, coeff=1.0, param_indices=None):
        """
        Create a new LinearlyParameterizedElementTerm

        Parameters
        ----------
        coeff : float, optional
            The term's coefficient

        param_indices : list
            A list of integers, specifying which parameters are muliplied
            together (and finally, with `coeff`) to form this term.
        """
        if param_indices is None:
            param_indices = []
        self.coeff = coeff
        self.paramIndices = param_indices


class LinearlyParamArbitraryOp(_DenseOperator):
    """
    An operation matrix parameterized such that each element depends only linearly on any parameter.

    Parameters
    ----------
    basematrix : numpy array
        a square 2D numpy array that acts as the starting point when
        constructin the operation's matrix.  The shape of this array sets
        the dimension of the operation.

    parameter_array : numpy array
        a 1D numpy array that holds the all the parameters for this
        operation.  The shape of this array sets is what is returned by
        `value_dimension(...)`.

    parameter_to_base_indices_map : dict
        A dictionary with keys == index of a parameter
        (i.e. in parameter_array) and values == list of 2-tuples
        indexing potentially multiple operation matrix coordinates
        which should be set equal to this parameter.

    left_transform : numpy array or None, optional
        A 2D array of the same shape as basematrix which left-multiplies
        the base matrix after parameters have been evaluated.  Defaults to
        no transform_inplace.

    right_transform : numpy array or None, optional
        A 2D array of the same shape as basematrix which right-multiplies
        the base matrix after parameters have been evaluated.  Defaults to
        no transform_inplace.

    real : bool, optional
        Whether or not the resulting operation matrix, after all
        parameter evaluation and left & right transforms have
        been performed, should be real.  If True, ValueError will
        be raised if the matrix contains any complex or imaginary
        elements.

    basis : Basis or {'pp','gm','std'} or None
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.  If None, certain functionality,
        such as access to Kraus operators, will be unavailable.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, base_matrix, parameter_array, parameter_to_base_indices_map, left_transform=None,
                 right_transform=None, real=False, basis=None, evotype="default", state_space=None):

        base_matrix = _np.array(_LinearOperator.convert_to_matrix(base_matrix), 'complex')
        #complex, even if passed all real base matrix

        elementExpressions = {}
        for p, ij_tuples in parameter_to_base_indices_map.items():
            for i, j in ij_tuples:
                assert((i, j) not in elementExpressions)  # only one parameter allowed per base index pair
                elementExpressions[(i, j)] = [LinearlyParameterizedElementTerm(1.0, [p])]

        typ = "d" if real else "complex"
        mx = _np.empty(base_matrix.shape, typ)
        self.baseMatrix = base_matrix
        self.parameterArray = parameter_array
        self.numParams = len(parameter_array)
        self.elementExpressions = elementExpressions
        assert(_np.isrealobj(self.parameterArray)), "Parameter array must be real-valued!"

        I = _np.identity(self.baseMatrix.shape[0], 'd')  # LinearlyParameterizedGates are currently assumed to be real
        self.leftTrans = left_transform if (left_transform is not None) else I
        self.rightTrans = right_transform if (right_transform is not None) else I
        self.enforceReal = real

        #Note: dense op reps *always* own their own data so setting writeable flag is OK
        _DenseOperator.__init__(self, mx, basis, evotype, state_space)
        self._ptr.flags.writeable = False  # only _construct_matrix can change array
        self._construct_matrix()  # construct base from the parameters

    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
        """
        matrix = self.baseMatrix.copy()
        for (i, j), terms in self.elementExpressions.items():
            for term in terms:
                param_prod = _np.prod([self.parameterArray[p] for p in term.paramIndices])
                matrix[i, j] += term.coeff * param_prod
        matrix = _np.dot(self.leftTrans, _np.dot(matrix, self.rightTrans))

        if self.enforceReal:
            if _np.linalg.norm(_np.imag(matrix)) > IMAG_TOL:
                raise ValueError("Linearly parameterized matrix has non-zero"
                                 "imaginary part (%g)!" % _np.linalg.norm(_np.imag(matrix)))
            matrix = _np.real(matrix)

        #Note: dense op reps *always* own their own data so setting writeable flag is OK
        assert(matrix.shape == (self.dim, self.dim))
        self._ptr.flags.writeable = True
        self._ptr[:, :] = matrix
        self._ptr.flags.writeable = False

    def _construct_param_to_base_indices_map(self):
        # build mapping for constructor, which has integer keys so ok for serialization
        param_to_base_indices_map = {}
        for (i, j), term in self.elementExpressions:
            assert(len(term.paramIndices) == 1)
            p = term.paramIndices[0]
            if p not in param_to_base_indices_map:
                param_to_base_indices_map[p] = []
            param_to_base_indices_map[p].append((i, j))
        return param_to_base_indices_map

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        param_to_base_indices_map = self._construct_param_to_base_indices_map()

        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['base_matrix'] = self._encodemx(self.baseMatrix)
        mm_dict['parameter_array'] = self._encodemx(self.parameterArray)
        mm_dict['parameter_to_base_indices_map'] = param_to_base_indices_map
        mm_dict['left_transform'] = self._encodemx(self.leftTrans)
        mm_dict['right_transform'] = self._encodemx(self.rightTrans)
        mm_dict['enforce_real'] = self.enforceReal

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        base_matrix = cls._decodemx(mm_dict['base_matrix'])
        parameter_array = cls._decodemx(mm_dict['parameter_array'])
        left_transform = cls._decodemx(mm_dict['left_transform'])
        right_transform = cls._decodemx(mm_dict['right_transform'])
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])

        return cls(base_matrix, parameter_array, mm_dict['parameter_to_base_indices_map'],
                   left_transform, right_transform, mm_dict['enforce_real'], mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return ((self.baseMatrix.shape == other.baseMatrix.shape)
                and _np.allclose(self.baseMatrix, other.baseMatrix, rtol=rtol, atol=atol)
                and _np.allclose(self.leftTrans, other.leftTrans, rtol=rtol, atol=atol)
                and _np.allclose(self.rightTrans, other.rightTrans, rtol=rtol, atol=atol)
                and (self._construct_param_to_base_indices_map() == other._construct_param_to_base_indices_map())
                and (self.enforceReal == other.enforceReal))

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.numParams

    def to_vector(self):
        """
        Extract a vector of the underlying operation parameters from this operation.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.parameterArray

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        self.parameterArray[:] = v
        self._construct_matrix()
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        derivMx = _np.zeros((self.numParams, self.dim, self.dim), 'complex')
        for (i, j), terms in self.elementExpressions.items():
            for term in terms:
                params_to_mult = [self.parameterArray[p] for p in term.paramIndices]
                for k, p in enumerate(term.paramIndices):
                    param_partial_prod = _np.prod(params_to_mult[0:k] + params_to_mult[k + 1:])  # exclude k-th factor
                    derivMx[p, i, j] += term.coeff * param_partial_prod

        derivMx = _np.dot(self.leftTrans, _np.dot(derivMx, self.rightTrans))  # (d,d) * (P,d,d) * (d,d) => (d,P,d)
        derivMx = _np.rollaxis(derivMx, 1, 3)  # now (d,d,P)
        derivMx = derivMx.reshape([self.dim**2, self.numParams])  # (d^2,P) == final shape

        if self.enforceReal:
            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)
            derivMx = _np.real(derivMx)

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
        return False

    def __str__(self):
        s = "Linearly Parameterized operation with shape %s, num params = %d\n" % \
            (str(self._ptr.shape), self.numParams)
        s += _mt.mx_to_string(self._ptr, width=5, prec=1)
        s += "\nParameterization:"
        for (i, j), terms in self.elementExpressions.items():
            tStr = ' + '.join(['*'.join(["p%d" % p for p in term.paramIndices])
                               for term in terms])
            s += "LinearOperator[%d,%d] = %s\n" % (i, j, tStr)
        return s
