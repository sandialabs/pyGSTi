"""
Defines the CPTPInstrument class
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
from pygsti.pgtypes import SpaceT
from pygsti.baseobjs.basis import Basis
from pygsti.modelmembers.povms.effect import POVMEffect
from pygsti.modelmembers.operations import LinearOperator
from pygsti.tools import optools as _ot

from typing import Union
BasisLike = Union[Basis, str]


class RootConjOperator(LinearOperator):
    """
    A linear operator parameterized by a matrix E where 0 ≤ E ≤ 1, whose action takes ρ to E½ ρ E½ .
    
    Every CPTR map can be obtained by pre-composing a CPTP map with this kind of linear operator.
    """

    EIGTOL_WARNING = 1e-10
    EIGTOL_ERROR   = 1e-8

    def __init__(self, effect: POVMEffect, basis: BasisLike):
        self._basis        = Basis.cast(basis, effect.dim)
        self._effect       = effect
        self._state_space  = effect.state_space
        self._evotype      = effect.evotype

        dim = self._state_space.dim
        self._rep = self._evotype.create_dense_superop_rep( _np.zeros((dim, dim)), self._basis, self._state_space )
        self._update_rep_base()
        LinearOperator.__init__(self, self._rep, self._evotype)
        self.init_gpindices()
    
    def submembers(self):
        return [self._effect]
    
    def _update_rep_base(self):
        # This function is directly analogous to TPInstrumentOp._construct_matrix.
        self._rep.base.flags.writeable = True
        assert(self._rep.base.shape == (self.dim, self.dim))
        effect_superket = self._effect.to_dense()
        mx = _ot.rootconj_superop(effect_superket, self._basis)
        self._rep.base[:] = mx
        self._rep.base.flags.writeable = False
        self._rep.base_has_changed()
        return
    
    def deriv_wrt_params(self, wrt_filter=None):
        return LinearOperator.deriv_wrt_params(self, wrt_filter)
    
    def has_nonzero_hessian(self):
        # This is not affine in its parameters.
        return True

    def from_vector(self, v, close=False, dirty_value=True):
        for sm, local_inds in zip(self.submembers(), self._submember_rpindices):
            sm.from_vector(v[local_inds], close, dirty_value)
        self._update_rep_base()
        return
    
    @property
    def num_params(self):
        return len(self.gpindices_as_array())

    def to_vector(self):
        v = _np.empty(self.num_params, 'd')
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            v[local_inds] = param_op.to_vector()
        return v

    def to_dense(self, on_space: SpaceT = 'HilbertSchmidt') -> _np.ndarray:
        assert on_space in ('HilbertSchmidt', 'minimal')
        out = self._rep.base.copy()
        out.flags.writeable = True
        return out


class SummedOperator(LinearOperator):


    def __init__(self, operators, basis: BasisLike):
        op = operators[0]
        self._basis        = Basis.cast(basis, op.dim)
        self._operators    = operators
        self._state_space  = op.state_space
        self._evotype      = op.evotype
        self._subreps      = [op._rep for op in self._operators]
        self._rep = self._evotype.create_sum_rep( self._subreps, self._state_space )
        LinearOperator.__init__(self, self._rep, self._evotype)
        self.init_gpindices()
        # NOTE: This class doesn't have a function analogous to _update_rep_base
        # that we use in RootConjOperator. We can get away with not having such
        # a function because it's the responsibility of op.from_vector(...)
        # to update op's attached OpRep.
        return
    
    def submembers(self):
        out = []
        hit = set()
        for op in self._operators:
            temp = op.submembers()
            for sm in temp:
                if id(temp) not in hit:
                    hit.add(id(temp))
                    out.append(sm)
        return out
    
    def deriv_wrt_params(self, wrt_filter=None):
        raise NotImplementedError()
    
    def has_nonzero_hessian(self):
        return any(op.has_nonzero_hession() for op in self._operators)

    def from_vector(self, v, close=False, dirty_value=True):
        for sm, local_inds in zip(self.submembers(), self._submember_rpindices):
            sm.from_vector(v[local_inds], close, dirty_value)
        return
    
    @property
    def num_params(self):
        return len(self.gpindices_as_array())

    def to_vector(self):
        v = _np.empty(self.num_params, 'd')
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            v[local_inds] = param_op.to_vector()
        return v

    def to_dense(self, on_space: SpaceT = 'HilbertSchmidt') -> _np.ndarray:
        assert on_space in ('HilbertSchmidt', 'minimal')
        on_space = 'HilbertSchmidt'
        out = self._operators[0].to_dense(on_space)
        if not out.flags.writeable:
            out = out.copy()
        for op in self._operators[1:]:
            temp = op.to_dense(on_space)
            out += temp
        return out
