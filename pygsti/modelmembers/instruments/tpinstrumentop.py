"""
The TPInstrumentOp class and supporting functionality.
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

from pygsti.modelmembers import modelmember as _mm
from pygsti.modelmembers.operations import DenseOperator as _DenseOperator
from pygsti.tools import slicetools as _slct


class TPInstrumentOp(_DenseOperator):
    """
    An element of a :class:`TPInstrument`.

    A partial implementation of :class:`LinearOperator` which encapsulates an
    element of a :class:`TPInstrument`.  Instances rely on their parent being a
    `TPInstrument`.

    Parameters
    ----------
    param_ops : list of LinearOperator objects
        A list of the underlying operation objects which constitute a simple
        parameterization of a :class:`TPInstrument`.  Namely, this is
        the list of [MT,D1,D2,...Dn] operations which parameterize *all* of the
        `TPInstrument`'s elements.

    index : int
        The index indicating which element of the `TPInstrument` the
        constructed object is.  Must be in the range
        `[0,len(param_ops)-1]`.
    """

    def __init__(self, param_ops, index):
        """
        Initialize a TPInstrumentOp object.

        Parameters
        ----------
        param_ops : list of LinearOperator objects
            A list of the underlying operation objects which constitute a simple
            parameterization of a :class:`TPInstrument`.  Namely, this is
            the list of [MT,D1,D2,...Dn] operations which parameterize *all* of the
            `TPInstrument`'s elements.

        index : int
            The index indicating which element of the `TPInstrument` the
            constructed object is.  Must be in the range
            `[0,len(param_ops)-1]`.
        """
        self.index = index
        self.num_instrument_elements = len(param_ops)
        _DenseOperator.__init__(self, _np.identity(param_ops[0].dim, 'd'), param_ops[0].evotype,
                                param_ops[0].state_space)  # Note: sets self.gpindices; TP assumed real

        #Set our own parent and gpindices based on param_ops
        # (this breaks the usual paradigm of having the parent object set these,
        #  but the exception is justified b/c the parent has set these members
        #  of the underlying 'param_ops' operations)
        dependents = [0, index + 1] if index < len(param_ops) - 1 \
            else list(range(len(param_ops)))
        self.relevant_param_ops = [param_ops[i] for i in dependents]

        self._construct_matrix()
        self.init_gpindices()

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.relevant_param_ops

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

        mm_dict['instrument_member_index'] = self.index
        mm_dict['number_of_instrument_elements'] = self.num_instrument_elements

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        index = mm_dict['instrument_member_index']
        nEls = mm_dict['number_of_instrument_elements']
        dependents = [0, index + 1] if index < nEls - 1 \
            else list(range(nEls))

        param_ops = [None] * nEls
        for subm_serial_id, paramop_index in zip(mm_dict['submembers'], dependents):
            param_ops[paramop_index] = serial_memo[subm_serial_id]

        return cls(param_ops, index)

    def _construct_matrix(self):
        """
        Mi = Di + MT for i = 1...(n-1)
           = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)
        """
        nEls = self.num_instrument_elements
        self._ptr.flags.writeable = True
        if self.index < nEls - 1:
            self._ptr[:, :] = _np.asarray(self.relevant_param_ops[1]  # i.e. param_ops[self.index + 1]
                                          + self.relevant_param_ops[0])  # i.e. param_ops[0]
        else:
            assert(self.index == nEls - 1), \
                "Invalid index %d > %d" % (self.index, nEls - 1)
            self._ptr[:, :] = _np.asarray(-sum(self.relevant_param_ops)  # all instrument param_ops == relevant
                                          - (nEls - 3) * self.relevant_param_ops[0])

        assert(self._ptr.shape == (self.dim, self.dim))
        self._ptr.flags.writeable = False
        self._ptr_has_changed()

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized
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
        Np = self.num_params
        derivMx = _np.zeros((self.dim**2, Np), 'd')
        Nels = self.num_instrument_elements

        off = 0
        if self.index < Nels - 1:  # matrix = Di + MT = param_ops[index+1] + param_ops[0]
            for i in [0, 1]:  # i.e. for param_ops [0, self.index + 1]
                Np = self.relevant_param_ops[i].num_params
                derivMx[:, off:off + Np] = self.relevant_param_ops[i].deriv_wrt_params()
                off += Np

        else:  # matrix = -(nEls-2)*MT-sum(Di), and relevant_param_ops == instrument's param_ops
            Np = self.relevant_param_ops[0].num_params
            derivMx[:, off:off + Np] = -(Nels - 2) * self.relevant_param_ops[0].deriv_wrt_params()
            off += Np

            for i in range(1, Nels):
                Np = self.relevant_param_ops[i].num_params
                derivMx[:, off:off + Np] = -self.relevant_param_ops[i].deriv_wrt_params()
                off += Np

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

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the operation parameters as an array of values.

        Returns
        -------
        numpy array
            The operation parameters as a 1D array with length num_params().
        """
        v = _np.empty(self.num_params, 'd')
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            v[local_inds] = param_op.to_vector()
        return v

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
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            param_op.from_vector(v[local_inds], close, dirty_value)

        #I dont' think this is still true, though the above, and from_vector in general, may benefit from a memo arg.
        # #Rely on the Instrument ordering of its elements: if we're being called
        # # to init from v then this is within the context of a TPInstrument's operations
        # # having been simplified and now being initialized from a vector (within a
        # # calculator).  We rely on the Instrument elements having their
        # # from_vector() methods called in self.index order.
        # if self.index < len(self.param_ops) - 1:  # final element doesn't need to init any param operations
        #     for i in self.dependents:  # re-init all my dependents (may be redundant)
        #         if i == 0 and self.index > 0: continue  # 0th param-operation already init by index==0 element
        #         paramop_local_inds = _mm._decompose_gpindices(
        #             self.gpindices, self.param_ops[i].gpindices)
        #         self.param_ops[i].from_vector(v[paramop_local_inds], close, dirty_value)

        self._construct_matrix()
