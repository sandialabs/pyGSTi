#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


"""
The classes in this file help us represent CPTR (completely positive trace-reducing)
channels using the error generator formalism.

By *trace reducing*, we formally mean *trace non-increasing*.

Definition
----------
A map is CPTR if and only if it has Kraus operators {K_i} where sum_i K_i^† K_i ≤ I.

Method
------
Let K_i = u p^½ be the polar decomposition of K_i. (We denote the psd factor from the
polar decomposition by p^½ instead of p because a K_i is *itself* a type of square-root.)

The Kraus-rank-1 channel σ ↦ (K_i) σ (K_i)^† can be represented as a composition of two
linear maps:

    1. A "root-conj" operator σ ↦ p^½ σ p^½, parameterized by the psd matrix p, and
    2. A standard unitary evolution, σ ↦ u σ u^†, parameterized by u.

pyGSTi has long had dozens of ways of representing (noisy) unitary evolution. This
file adds classes to represent the root-conj part of a CPTR map, and to sum multiple
Kraus-rank-1 terms together when needed.
"""
from __future__ import annotations

from typing import Tuple, Any, TYPE_CHECKING
if TYPE_CHECKING:
    import torch as _torch
try:
    import torch as _torch
except ImportError:
    pass

import numpy as _np
from pygsti.pgtypes import SpaceT
from pygsti.baseobjs.basis import Basis, BuiltinBasis as _BuiltinBasis
from pygsti.modelmembers.povms.effect import POVMEffect
from pygsti.modelmembers.operations.linearop import LinearOperator
from pygsti.modelmembers.torchable import Torchable as _Torchable
from pygsti.tools import optools as _ot

from typing import Union
BasisLike = Union[Basis, str]


class RootConjOperator(LinearOperator, _Torchable):
    """
    Represents a linear operator that acts as ρ ↦ E^½ ρ E^½, where E is a
    Hermitian matrix represented by a POVMEffect object.
    
    We need 0 ≤ E ≤ I for this operator to be well-defined. This is gauranteed
    by some POVMEffect subclasses, but not all. The RootConjOperator will raise
    an error if E's eigenvalues fall too far outside the range [0, 1].

    Parameters
    ----------
    effect : POVMEffect
        A POVM effect whose superket encodes the PSD matrix E. This linear
        operator's gpindices are shared with those of E.
    
    basis : Basis or str
        The operator basis in which the superoperator is represented.

    Class attributes
    ----------------
    EIGTOL_WARNING : float
        How far an eigenvalue of E may fall outside [0, 1] before a warning is issued when
        forming E½.  Defaults to :data:`~pygsti.tools.optools.EFFECT_EIGVAL_ABSTOL_WARN`.
    EIGTOL_ERROR : float
        How far an eigenvalue of E may fall outside [0, 1] before an error is raised.
        Defaults to :data:`~pygsti.tools.optools.EFFECT_EIGVAL_ABSTOL_ERROR`.
    """

    EIGTOL_WARNING = _ot.EFFECT_EIGVAL_ABSTOL_WARN
    EIGTOL_ERROR   = _ot.EFFECT_EIGVAL_ABSTOL_ERROR

    def __init__(self, effect: POVMEffect, basis: BasisLike):
        self._basis        = Basis.cast(basis, effect.dim)
        self._effect       = effect
        self._state_space  = effect.state_space
        self._evotype      = effect.evotype

        dim = self._state_space.dim
        self._rep = self._evotype.create_dense_superop_rep(
            _np.zeros((dim, dim)), self._basis, self._state_space
        )
        self._update_rep_base()
        LinearOperator.__init__(self, self._rep, self._evotype)
        self.init_gpindices()

    def submembers(self) -> list[POVMEffect]:
        return [self._effect]

    def to_memoized_dict(self, mmg_memo: dict) -> dict:
        mm_dict = super().to_memoized_dict(mmg_memo)
        mm_dict['basis'] = self._basis.to_nice_serialization()
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict: dict, serial_memo: dict) -> RootConjOperator:
        effect = serial_memo[mm_dict['submembers'][0]]
        basis = Basis.from_nice_serialization(mm_dict['basis'])
        return cls(effect, basis)

    def _update_rep_base(self) -> None:
        # Analogous to TPInstrumentOp._construct_matrix.
        self._rep.base.flags.writeable = True
        assert(self._rep.base.shape == (self.dim, self.dim))
        effect_superket = self._effect.to_dense()
        mx = _ot.rootconj_superop(effect_superket, self._basis, self.EIGTOL_WARNING, self.EIGTOL_ERROR)
        self._rep.base[:] = mx
        self._rep.base.flags.writeable = False
        self._rep.base_has_changed()

    def deriv_wrt_params(self, wrt_filter: _np.ndarray | list | None = None) -> _np.ndarray:
        return LinearOperator.deriv_wrt_params(self, wrt_filter)

    def has_nonzero_hessian(self) -> bool:
        # Not affine in its parameters.
        return True

    def from_vector(self, v: _np.ndarray, close: bool = False, dirty_value: bool = True) -> None:
        for sm, local_inds in zip(self.submembers(), self._submember_rpindices):
            sm.from_vector(v[local_inds], close, dirty_value)
        self._update_rep_base()

    @property
    def num_params(self) -> int:
        return len(self.gpindices_as_array())

    def to_vector(self) -> _np.ndarray:
        v = _np.empty(self.num_params, 'd')
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            v[local_inds] = param_op.to_vector()
        return v

    def to_dense(self, on_space: SpaceT = 'HilbertSchmidt') -> _np.ndarray:
        assert on_space in ('HilbertSchmidt', 'minimal')
        out = self._rep.base.copy()
        out.flags.writeable = True
        return out

    def stateless_data(self, real_dtype: _torch.dtype, device: _torch.Device) -> Tuple[Any, ...]:
        """Bake the constants used by :meth:`torch_base` (see there for the degeneracy caveat).

        All free parameters belong to the wrapped effect; the basis element matrices ``B`` and the
        std<->basis change-of-basis matrices ``toMx`` / ``fromMx`` are constant and are pre-computed
        here (mirroring :func:`pygsti.tools.optools.rootconj_superop`).
        """
        cdtype = _torch.complex128 if real_dtype.itemsize == 8 else _torch.complex64
        dim = self._state_space.dim
        # B[i] = self._basis.elements[i]; E = einsum('i,iab->ab', effect_superket, B) == vec_to_stdmx.
        B = _torch.from_numpy(_np.array(self._basis.elements)).to(dtype=cdtype, device=device)
        # change_basis(mx_std, 'std', self._basis) == toMx @ mx_std @ fromMx (constant matrices).
        std_basis = _BuiltinBasis('std', dim)
        to_np = _np.ascontiguousarray(std_basis.create_transform_matrix(self._basis))
        toMx  = _torch.from_numpy(to_np).to(dtype=cdtype, device=device)
        from_np = _np.ascontiguousarray(self._basis.create_transform_matrix('std'))
        fromMx  = _torch.from_numpy(from_np).to(dtype=cdtype, device=device)
        return (type(self._effect), self._effect.stateless_data(real_dtype, device), B, toMx, fromMx)

    @staticmethod
    def torch_base(sd: Tuple[Any, ...], t_param: _torch.Tensor) -> _torch.Tensor:
        """Differentiable reconstruction of the ρ ↦ E½ ρ E½ superoperator (see :func:`rootconj_superop`).

        .. warning::

            This uses ``torch.linalg.eigh`` on the effect matrix ``E``.  The backward pass of ``eigh``
            contains ``1 / (λᵢ − λⱼ)`` factors, so the Jacobian returned by ``torch.func.jacrev`` is
            numerically unstable (and NaN in the exactly-degenerate limit) for effects with (near-)
            degenerate spectra -- e.g. ideal projective effects on two or more qubits.  The clamp of
            the eigenvalues to ``[0, 1]`` is also subgradient-ambiguous at eigenvalues of exactly 0 or
            1.  Use torch's reverse-mode AD for this operator only for effects whose spectra are interior
            to ``(0, 1)`` and non-degenerate; the value and forward-mode AD paths have no such restriction.

        Unlike the numpy reference :func:`~pygsti.tools.optools.rootconj_superop` -- which warns
        when an eigenvalue of ``E`` strays outside ``[0, 1]`` by more than :attr:`EIGTOL_WARNING`
        and raises a ``ValueError`` past :attr:`EIGTOL_ERROR` -- this path clamps silently at any
        magnitude of violation.  So under torch-driven optimization it can keep returning
        (clamped) values in regions where the numpy path would refuse to evaluate.
        """
        etype, esd, B, toMx, fromMx = sd
        v = etype.torch_base(esd, t_param).to(B.dtype)  # whole t_param: params delegate to the effect
        E = _torch.einsum('i,iab->ab', v, B)            # vec_to_stdmx
        E = (E + E.mH) / 2                               # match eigendecomposition's Hermitian symmetrize
        vals, V = _torch.linalg.eigh(E)
        R = (V * _torch.sqrt(_torch.clamp(vals, 0.0, 1.0)).unsqueeze(0)) @ V.mH
        # plain (non-conjugate) transpose, matching np.kron(R, R.T); .contiguous() because
        # torch.kron cannot view the transposed (non-contiguous) operand.
        return (toMx @ _torch.kron(R, R.transpose(0, 1).contiguous()) @ fromMx).real


class SummedOperator(LinearOperator):
    """
    A linear operator whose superoperator is the sum of a list of constituent operators.

    This class represents a map as a sum of constituent maps -- for example a
    multi-Kraus-term CPTR map written as a sum of single-Kraus-term summands.  It is
    retained as a tested general-purpose building block, although it is not used
    elsewhere in pyGSTi itself.

    The parameter vector is the concatenation (with shared-index deduplication via
    gpindices) of the submembers' parameter vectors.  The constituent operators' `_rep`
    objects are linked directly, so calling `from_vector` on a submember automatically
    updates this operator's representation.

    Parameters
    ----------
    operators : list of LinearOperator
        The operators to sum.  All must share the same evotype, state space, and
        superoperator dimension.
    basis : Basis or str
        The operator basis (used for bookkeeping; the operators themselves must
        already be expressed in this basis).

    Notes
    -----
    `deriv_wrt_params` is not implemented for this class.
    """

    def __init__(self, operators: list[LinearOperator], basis: BasisLike):
        op = operators[0]
        self._basis        = Basis.cast(basis, op.dim)
        self._operators    = operators
        self._state_space  = op.state_space
        self._evotype      = op.evotype
        self._subreps      = [op._rep for op in self._operators]
        self._rep = self._evotype.create_sum_rep(self._subreps, self._state_space)
        LinearOperator.__init__(self, self._rep, self._evotype)
        self.init_gpindices()
        # NOTE: No _update_rep_base analogue is needed here.  Each constituent
        # operator's from_vector(...) updates its own attached OpRep, and the
        # sum rep reflects those changes automatically.

    def submembers(self) -> list:
        out = []
        hit = set()
        for op in self._operators:
            temp = op.submembers()
            for sm in temp:
                if id(temp) not in hit:
                    hit.add(id(temp))
                    out.append(sm)
        return out

    def deriv_wrt_params(self, wrt_filter: _np.ndarray | list | None = None) -> _np.ndarray:
        raise NotImplementedError()

    def has_nonzero_hessian(self) -> bool:
        return any(op.has_nonzero_hessian() for op in self._operators)

    def from_vector(self, v: _np.ndarray, close: bool = False, dirty_value: bool = True) -> None:
        for sm, local_inds in zip(self.submembers(), self._submember_rpindices):
            sm.from_vector(v[local_inds], close, dirty_value)

    @property
    def num_params(self) -> int:
        return len(self.gpindices_as_array())

    def to_vector(self) -> _np.ndarray:
        v = _np.empty(self.num_params, 'd')
        for param_op, local_inds in zip(self.submembers(), self._submember_rpindices):
            v[local_inds] = param_op.to_vector()
        return v

    def to_dense(self, on_space: SpaceT = 'HilbertSchmidt') -> _np.ndarray:
        assert on_space in ('HilbertSchmidt', 'minimal')
        out = self._operators[0].to_dense('HilbertSchmidt')
        if not out.flags.writeable:
            out = out.copy()
        for op in self._operators[1:]:
            out += op.to_dense('HilbertSchmidt')
        return out
