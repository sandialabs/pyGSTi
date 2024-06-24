"""
The TPState class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import torch as _torch
try:
    import torch as _torch
except ImportError:
    pass

import numpy as _np
from pygsti.baseobjs import Basis as _Basis
from pygsti.baseobjs import statespace as _statespace
from pygsti.modelmembers.torchable import Torchable as _Torchable
from pygsti.modelmembers.states.densestate import DenseState as _DenseState
from pygsti.modelmembers.states.state import State as _State
from pygsti.baseobjs.protectedarray import ProtectedArray as _ProtectedArray


class TPState(_DenseState, _Torchable):
    """
    A fixed-unit-trace state vector.

    This state vector is fully parameterized except for the first element, which
    is frozen to be `1/(d**0.25)`.  This is so that, when the state vector is
    interpreted in the Pauli or Gell-Mann basis, the represented density matrix
    has trace == 1.  This restriction is frequently used in conjuction with
    trace-preserving (TP) gates, hence its name.

    Parameters
    ----------
    vec : array_like or State
        a 1D numpy array representing the state.  The
        shape of this array sets the dimension of the state.

    basis : Basis or str
        The basis that `vec` is in.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    #Note: here we assume that the first basis element is (1/sqrt(x) * I),
    # where I the d-dimensional identity (where len(vector) == d**2). So
    # if Tr(basisEl*basisEl) == Tr(1/x*I) == d/x must == 1, then we must
    # have x == d.  Thus, we multiply this first basis element by
    # alpha = 1/sqrt(d) to obtain a trace-1 matrix, i.e., finding alpha
    # s.t. Tr(alpha*[1/sqrt(d)*I]) == 1 => alpha*d/sqrt(d) == 1 =>
    # alpha = 1/sqrt(d) = 1/(len(vec)**0.25).
    def __init__(self, vec, basis=None, evotype="default", state_space=None):
        vector = _State._to_vector(vec)
        if basis is None:
            dim = vector.size
            basis = 'pp' if int(2**_np.log2(dim)) == dim else 'gm'
        _DenseState.__init__(self, vector, basis, evotype, state_space)
        basis = self._basis              # <-- __init__ ensures that self._basis is a Basis object.
        firstEl = basis.elsize ** -0.25  # <-- not dim, as the dimension of the vector space may be less
        if not _np.isclose(vector[0], firstEl):
            raise ValueError("Cannot create TPState: first element must equal %g!" % firstEl)
        assert(isinstance(self.columnvec, _ProtectedArray))
        self._paramlbls = _np.array(["VecElement %d" % i for i in range(1, self.dim)], dtype=object)

    @property
    def columnvec(self):
        """
        Direct access the the underlying data as column vector, i.e, a (dim,1)-shaped array.
        """
        bv = self._ptr.view()
        bv.shape = (bv.size, 1)
        return _ProtectedArray(bv, indices_to_protect=(0, 0))

    def set_dense(self, vec):
        """
        Set the dense-vector value of this state vector.

        Attempts to modify this state vector's parameters so that the raw
        state vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or State
            A numpy array representing a state vector, or a State object.

        Returns
        -------
        None
        """
        vec = _State._to_vector(vec)
        firstEl = (self.dim)**-0.25
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        if not _np.isclose(vec[0], firstEl):
            raise ValueError("Cannot create TPState: "
                             "first element must equal %g!" % firstEl)
        self._ptr[1:] = vec[1:]
        self._ptr_has_changed()
        self.dirty = True

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.dim - 1

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self._ptr[1:]  # .real in case of complex matrices?

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
        #assert(_np.isclose(self._ptr[0], (self.dim)**-0.25))  # takes too much time!
        self._ptr[1:] = v
        self._ptr_has_changed()
        self.dirty = dirty_value

    def stateless_data(self) -> Tuple[_torch.Tensor]:
        dim = self.dim
        t_const = (dim ** -0.25) * _torch.ones(1, dtype=_torch.double) 
        return (t_const,)

    @staticmethod
    def torch_base(sd: Tuple[_torch.Tensor], t_param: _torch.Tensor) -> _torch.Tensor:
        t_const = sd[0]
        t = _torch.concat((t_const, t_param)) 
        return t

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
        derivMx = _np.identity(self.dim, 'd')  # TP vecs assumed real
        derivMx = derivMx[:, 1:]  # remove first col ( <=> first-el parameters )
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

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        vec = cls._decodemx(mm_dict['dense_superket_vector'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        basis = _Basis.from_nice_serialization(mm_dict['basis']) if (mm_dict['basis'] is not None) else None

        # HACK -- REMOVE LATER -- allows loading objs saved before fixing TP state bug
        # at commit aa8a29049a9c3ddafe598d903602ab5afdb7aad4
        if mm_dict['basis'] is not None and 'name' in mm_dict['basis'] and mm_dict['basis']['name'] == "*Empty*":
            basis = None

        return cls(vec, basis, mm_dict['evotype'], state_space)  # use basis=None to skip 1st element check
