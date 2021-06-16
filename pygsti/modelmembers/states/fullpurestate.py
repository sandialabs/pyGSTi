"""
The FullPureState class and supporting functionality.
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

from pygsti.modelmembers.states.densestate import DensePureState as _DensePureState


class FullPureState(_DensePureState):
    """
    A "fully parameterized" state vector where each element is an independent parameter.

    Parameters
    ----------
    vec : array_like or State
        a 1D numpy array representing the state operation.  The
        shape of this array sets the dimension of the state op.

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-ket.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, purevec, basis="pp", evotype="default", state_space=None):
        _DensePureState.__init__(self, purevec, basis, evotype, state_space)
        self._paramlbls = _np.array(["VecElement Re(%d)" % i for i in range(self.state_space.udim)]
                                    + ["VecElement Im(%d)" % i for i in range(self.state_space.udim)], dtype=object)

    #REMOVE
    #Cannot set to arbitrary vector
    #def set_dense(self, vec):
    #    """
    #    Set the dense-vector value of this SPAM vector.
    #
    #    Attempts to modify this SPAM vector's parameters so that the raw
    #    SPAM vector becomes `vec`.  Will raise ValueError if this operation
    #    is not possible.
    #
    #    Parameters
    #    ----------
    #    vec : array_like or State
    #        A numpy array representing a SPAM vector, or a State object.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    vec = State._to_vector(vec)
    #    if(vec.size != self.dim):
    #        raise ValueError("Argument must be length %d" % self.dim)
    #    self._ptr[:] = vec
    #    self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 2 * self.state_space.udim

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        #TODO: what if _base_1d isn't implemented - use init_from_dense_purevec?
        return _np.concatenate((self._ptr.real, self._ptr.imag), axis=0)

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the state vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of state vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this state vector's current
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
        self._ptr[:] = v[0:self.state_space.udim] + 1j * v[self.state_space.udim:]
        self._ptr_has_changed()
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this state vector.

        Construct a matrix whose columns are the derivatives of the state vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per state vector parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        derivMx = _np.concatenate((_np.identity(self.state_space.udim, complex),
                                   1j * _np.identity(self.state_space.udim, complex)), axis=1)
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this state vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return False
