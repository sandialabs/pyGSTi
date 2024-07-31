"""
The MCMInstrumentOp class and supporting functionality.
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

from pygsti.modelmembers.operations import DenseOperator as _DenseOperator

class MCMInstrumentOp(_DenseOperator):
    """
    An element of a :class:`MCMInstrument`.

    A partial implementation of :class:`LinearOperator` which encapsulates an
    element of a :class:`MCMInstrument`.  Instances rely on their parent being a
    `MCMInstrument`.

    Parameters
    ----------
    noise_map : ExpErrorgenOp
        The noise map in the auxiliary picture which parameterize *all* of the
        `MCMInstrument`'s elements.

    index : int
        The index indicating which element of the `TPInstrument` the
        constructed object is.  Must be in the range
        `[0,len(param_ops)-1]`.

    basis : Basis or {'pp','gm','std'} or None
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.  If None, certain functionality,
        such as access to Kraus operators, will be unavailable.
        
    right_isometry : numpy array
        The right isometry to be used to map from the auxiliary picture to the submembers. 
        Should be inherited from base `MCMInstrument`.

    left_isometry : list of numpy arrays
        The right isometry to be used to map from the auxiliary picture to the submembers.
        Should be inherited from base `MCMInstrument` and will have same number of elements
        as 'index.' 
    
    """
    def __init__(self, noise_map, index, right_isometry, left_isometry, basis=None):
        dim = 4 
        self.index = index
        self.noise_map = noise_map 
        
        self.op_right_iso = right_isometry 
        self.op_left_iso = left_isometry[index] 

        _DenseOperator.__init__(self, _np.identity(dim, 'd'), basis, self.noise_map.evotype,
                                _statespace.default_space_for_dim(dim))
        self._construct_matrix()
        self.init_gpindices()
        
    def _construct_matrix(self):
        self._ptr.flags.writeable = True
        
        self._ptr[:, :] = self.op_left_iso @ self.noise_map.to_dense() @ self.op_right_iso 
    
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

        noise_map_derivMx = self.noise_map.deriv_wrt_params()
        ptm_wrt_params = []
        for param_num in range(240):
            ptm_noise_map_derivMx = []
            for matrix_el in range(256):
                ptm_noise_map_derivMx += [noise_map_derivMx[matrix_el][param_num]]
            ptm_wrt_params += [list(_np.ravel(self.op_left_iso @ _np.reshape(ptm_noise_map_derivMx, (16,16)) @ self.op_right_iso))]
        derivMx = _np.array(ptm_wrt_params).transpose()
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

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

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.noise_map]