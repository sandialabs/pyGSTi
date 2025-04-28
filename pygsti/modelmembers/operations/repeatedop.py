"""
Defines the RepeatedOp class
"""
# ***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************

import numpy as _np
import scipy.sparse as _sps

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.evotypes import Evotype as _Evotype


class RepeatedOp(_LinearOperator):
    """
    An operation map that is the composition of a number of map-like factors (possibly other `LinearOperator`)

    Parameters
    ----------
    op_to_repeat : list
        A `LinearOperator`-derived object that is repeated
        some integer number of times to produce this operator.

    num_repetitions : int
        the power to exponentiate `op_to_exponentiate` to.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
        The special value `"auto"` uses the evolutio ntype of `op_to_repeat`.
    """

    def __init__(self, op_to_repeat, num_repetitions, evotype="auto"):
        #We may not actually need to save these, since they can be inferred easily
        self.repeated_op = op_to_repeat
        self.num_repetitions = num_repetitions

        state_space = op_to_repeat.state_space

        if evotype == "auto":
            evotype = op_to_repeat._evotype
        evotype = _Evotype.cast(evotype, state_space=state_space)
        rep = evotype.create_repeated_rep(self.repeated_op._rep, self.num_repetitions, state_space)
        _LinearOperator.__init__(self, rep, evotype)
        self.init_gpindices()  # initialize our gpindices based on sub-members

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.repeated_op]

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
        self.repeated_op.set_time(t)

    def to_sparse(self, on_space='minimal'):
        """
        Return the operation as a sparse matrix

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if self.num_repetitions == 0:
            return _sps.identity(self.dim, dtype=_np.dtype('d'), format='csr')

        op = self.repeated_op.to_sparse(on_space)
        mx = op.copy()
        for i in range(self.num_repetitions - 1):
            mx = mx.dot(op)
        return mx

    def to_dense(self, on_space='minimal'):
        """
        Return this operation as a dense matrix.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        op = self.repeated_op.to_dense(on_space)
        return _np.linalg.matrix_power(op, self.num_repetitions)

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.repeated_op.paramter_labels

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.repeated_op.num_params

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        return self.repeated_op.to_vector()

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
        assert(len(v) == self.num_params)
        self.repeated_op.from_vector(v, close, dirty_value)
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Constructs a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter. An
        empty 2D array in the StaticArbitraryOp case (num_params == 0).

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
        mx = self.repeated_op.to_dense(on_space='minimal')

        mx_powers = {0: _np.identity(self.dim, 'd'), 1: mx}
        for i in range(2, self.num_repetitions):
            mx_powers[i] = _np.dot(mx_powers[i - 1], mx)

        dmx = _np.transpose(self.repeated_op.deriv_wrt_params(wrt_filter))  # (num_params, dim^2)
        dmx.shape = (dmx.shape[0], self.dim, self.dim)  # set shape for multiplication below

        deriv = _np.zeros((self.dim, dmx.shape[0], self.dim), 'd')
        for k in range(1, self.num_repetitions + 1):
            #deriv += mx_powers[k-1] * dmx * mx_powers[self.num_repetitions-k]
            deriv += _np.dot(mx_powers[k - 1], _np.dot(dmx, mx_powers[self.num_repetitions - k]))
            #        (D,D) * ((P,D,D) * (D,D)) => (D,D) * (P,D,D) => (D,P,D)

        deriv = _np.moveaxis(deriv, 1, 2)
        deriv = deriv.reshape((self.dim**2, deriv.shape[2]))
        return deriv

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
        mm_dict = super().to_memoized_dict(mmg_memo)
        mm_dict['num_repetitions'] = self.num_repetitions
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        repeated_op = serial_memo[mm_dict['submembers'][0]]
        return cls(repeated_op, mm_dict['num_repetitions'], mm_dict['evotype'])

    def __str__(self):
        """ Return string representation """
        s = "Repeated operation that repeates the below op %d times\n" % self.num_repetitions
        s += str(self.repeated_op)
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "repeats %d times" % self.num_repetitions
