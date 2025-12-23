"""
Defines the ModelParamsInterposer class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


class ModelParamsInterposer(_NicelySerializable):
    """
    A function class that sits in between an :class:`OpModel`'s parameter vector and those of its operations.
    """
    def __init__(self, num_params, num_op_params):
        super().__init__()
        self.num_params = num_params
        self.num_op_params = num_op_params

    def model_paramvec_to_ops_paramvec(self, v):
        return v

    def ops_paramvec_to_model_paramvec(self, w):
        return w

    def ops_paramlbls_to_model_paramlbls(self, w):
        return w

    def deriv_op_params_wrt_model_params(self):
        assert(self.num_params == self.num_op_params)
        return _np.identity(self.num_params, 'd')

    def ops_params_dependent_on_model_params(self, model_param_indices):
        return _np.array(sorted(model_param_indices), _np.int64)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'num_model_params': self.num_params,
                      'num_op_params': self.num_op_params
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        return cls(state['num_model_params'], state['num_op_params'])


class LinearInterposer(ModelParamsInterposer):
    """
    Model parameters are linear combinations of operation parameters.

    FUTURE possibility:
    Also includes functionality for taking square/sqrt of parameters
    to be compatible with cases where the "true" parameter we want to
    include in the linear combination is the square of an actual parameter
    (to constrain it to being positive).
    """

    def __init__(self, transform_matrix):
        self.transform_matrix = transform_matrix  # cols specify a model parameter in terms of op params.
        self.inv_transform_matrix = _np.linalg.pinv(self.transform_matrix)
        self.inv_transform_matrix[_np.abs(self.inv_transform_matrix) < 1e-10] = 0
        self.embedder_matrix = _np.eye(self.transform_matrix.shape[1])
        self.full_span_transform_matrix = self.transform_matrix.copy()
        super().__init__(transform_matrix.shape[1], transform_matrix.shape[0])
    def model_paramvec_to_ops_paramvec(self, v):
        return self.transform_matrix @ v

    def ops_paramvec_to_model_paramvec(self, w):
        return self.inv_transform_matrix @ w

    def ops_paramlbls_to_model_paramlbls(self, wl):
        # This can and should be improved later - particularly this will be awful when labels (els of wl) are tuples.
        ret = []
        for irow in range(self.inv_transform_matrix.shape[0]):
            lbl = ' + '.join(["%g%s" % (coeff, str(lbl)) for coeff, lbl in zip(self.inv_transform_matrix[irow, :], wl) if abs(coeff)>1e-10])
            ret.append(lbl)
        return ret

    def deriv_op_params_wrt_model_params(self):
        return self.transform_matrix

    def ops_params_dependent_on_model_params(self, model_param_indices):
        """ TODO: docstring: returns the indices of op parameters that are influenced by the given model params"""
        op_param_indices = set()
        for j in model_param_indices:
            op_param_indices.update([i for i, el in enumerate(self.transform_matrix[:, j]) if el != 0.0])
        return _np.array(sorted(op_param_indices), _np.int64)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'transform_matrix': self._encodemx(self.transform_matrix)})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        return cls(cls._decodemx(state['transform_matrix']))

    def __eq__(self, other):
        if not isinstance(other, LinearInterposer):
            return False

        if self.transform_matrix.shape != other.transform_matrix.shape:
            return False
        
        return _np.allclose(self.transform_matrix, other.transform_matrix)