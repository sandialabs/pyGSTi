"""
The FullUnitaryOp class and supporting functionality.
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

from pygsti.modelmembers.operations.denseop import DenseUnitaryOperator as _DenseUnitaryOperator
from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot


class FullUnitaryOp(_DenseUnitaryOperator):
    """
    An operation matrix that is fully parameterized.

    That is, each element of the operation matrix is an independent parameter.

    Parameters
    ----------
    m : array_like or LinearOperator
        a square 2D array-like or LinearOperator object representing the operation action.
        The shape of m sets the dimension of the operation.

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, m, basis='pp', evotype="default", state_space=None):
        _DenseUnitaryOperator.__init__(self, m, basis, evotype, state_space)
        udim = self.state_space.udim  # or self._ptr.shape[0]
        self._paramlbls = _np.array(["MxElement Re(%d,%d)" % (i, j) for i in range(udim) for j in range(udim)]
                                    + ["MxElement Im(%d,%d)" % (i, j) for i in range(udim)
                                       for j in range(udim)], dtype=object)

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
        mx = _LinearOperator.convert_to_matrix(m)
        udim = self.state_space.udim  # maybe create a self.udim?
        if(mx.shape != (udim, udim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (udim, udim))
        self._ptr[:, :] = _np.array(mx)
        self._ptr_has_changed()
        self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 2 * self._ptr.size

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        # _np.concatenate will make a copy for us, so use ravel instead of flatten.
        return _np.concatenate((self._ptr.real.ravel(), self._ptr.imag.ravel()), axis=0)

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
        udim = self.state_space.udim  # maybe create a self.udim?
        assert(self._ptr.shape == (udim, udim))
        self._ptr[:, :] = v[0:udim**2].reshape((udim, udim)) + \
            1j * v[udim**2:].reshape((udim, udim))
        self._ptr_has_changed()
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
            Array of derivatives with shape (dimension^2, num_params)
        """
        udim = self.state_space.udim  # maybe create a self.udim?
        derivMx = _np.concatenate((_np.identity(udim**2, 'complex'),
                                   1j * _np.identity(udim**2, 'complex')),
                                  axis=1)
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

    def transform_inplace(self, s):
        """
        Update operation matrix `O` with `inv(s) * O * s`.

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        from pygsti.models import gaugegroup as _gaugegroup
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):

            #Just to this the brute force way for now - there should be a more elegant & faster way!
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse

            my_superop_mx = _ot.unitary_to_superop(self._ptr, self._basis)
            my_superop_mx = Uinv @ (my_superop_mx @ U)

            self._ptr[:, :] = _ot.superop_to_unitary(my_superop_mx, self._basis)
            self._ptr_has_changed()
            self.dirty = True
        else:
            raise ValueError("Invalid transform for this FullUnitaryOp: type %s" % str(type(s)))

    def spam_transform_inplace(self, s, typ):
        """
        Update operation matrix `O` with `inv(s) * O` OR `O * s`, depending on the value of `typ`.

        This functions as `transform_inplace(...)` but is used when this
        Lindblad-parameterized operation is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(s) * rho`, so `self -> inv(s) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * s`
        so that `self -> self * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).

        Returns
        -------
        None
        """
        assert(typ in ('prep', 'effect')), "Invalid `typ` argument: %s" % typ

        from pygsti.models import gaugegroup as _gaugegroup
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse

            #Just to this the brute force way for now - there should be a more elegant & faster way!
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse

            my_superop_mx = _ot.unitary_to_superop(self._ptr, self._basis)

            #Note: this code may need to be tweaked to work with sparse matrices
            if typ == "prep":
                my_superop_mx = Uinv @ my_superop_mx
            else:
                my_superop_mx = my_superop_mx @ U

            self._ptr[:, :] = _ot.superop_to_unitary(my_superop_mx, self._basis)
            self._ptr_has_changed()
            self.dirty = True
        else:
            raise ValueError("Invalid transform for this FullUnitaryOp: type %s" % str(type(s)))
