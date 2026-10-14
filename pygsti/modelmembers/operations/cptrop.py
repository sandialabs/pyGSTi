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
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


import numpy as _np
from pygsti.pgtypes import SpaceT
from pygsti.baseobjs.basis import Basis, BuiltinBasis as _BuiltinBasis
from pygsti.modelmembers.povms.effect import POVMEffect
from pygsti.modelmembers.operations.linearop import LinearOperator
from pygsti.modelmembers.torchable import Torchable as _Torchable
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot

from typing import Union
BasisLike = Union[Basis, str]


if _TORCH_AVAILABLE:
    class _EffectSqrt(_torch.autograd.Function):
        """
        E ↦ E½ for a Hermitian E whose eigenvalues are clamped to [0, 1], as in
        :func:`~pygsti.tools.optools.rootconj_superop`.

        Autograd through ``torch.linalg.eigh`` differentiates eigenvectors, which divides by
        eigenvalue gaps and gives NaN at repeated eigenvalues. This function instead applies the
        Fréchet derivative of the square root (see
        :func:`~pygsti.tools.matrixtools.sqrtm_frechet_derivative`), whose denominators are
        sums of square roots. That map is self-adjoint under the real Frobenius inner product,
        so the same formula serves for ``backward`` and ``jvp``, and ``generate_vmap_rule`` makes
        both ``torch.func.jacrev`` and ``torch.func.jacfwd`` work.

        Eigenvalues at or below `tol` are treated as zero, and the derivative's block on them is
        zero, as in :func:`~pygsti.tools.matrixtools.sqrtm_frechet_derivative`: the minimum-norm
        solution of the Sylvester equation. That keeps the map self-adjoint.
        """
        generate_vmap_rule = True

        @staticmethod
        def forward(E: _torch.Tensor, tol: float):
            vals, V = _torch.linalg.eigh(E)
            vals = _torch.clamp(vals, 0.0, 1.0)
            R = (V * _torch.sqrt(vals).to(V.dtype).unsqueeze(0)) @ V.mH
            return R, V, vals

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, tol = inputs
            _, V, vals = output
            ctx.mark_non_differentiable(V, vals)
            ctx.save_for_backward(V, vals)
            ctx.save_for_forward(V, vals)
            ctx.tol = tol

        @staticmethod
        def _frechet(V: _torch.Tensor, vals: _torch.Tensor, tol: float, H: _torch.Tensor) -> _torch.Tensor:
            roots = _torch.sqrt(_torch.where(vals > tol, vals, _torch.zeros_like(vals)))
            root_sums = roots.unsqueeze(1) + roots.unsqueeze(0)
            positive = root_sums > 0
            weights = _torch.where(positive, 1.0 / _torch.where(positive, root_sums, _torch.ones_like(root_sums)),
                                   _torch.zeros_like(root_sums)).to(V.dtype)
            return V @ ((V.mH @ H @ V) * weights) @ V.mH

        @staticmethod
        def backward(ctx, grad_R, grad_V, grad_vals):
            V, vals = ctx.saved_tensors
            return _EffectSqrt._frechet(V, vals, ctx.tol, grad_R), None

        @staticmethod
        def jvp(ctx, tangent_E, tangent_tol):
            V, vals = ctx.saved_tensors
            return _EffectSqrt._frechet(V, vals, ctx.tol, tangent_E), None, None


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
    DERIV_EIGTOL : float
        Eigenvalues of E at or below this are treated as zero by :meth:`deriv_wrt_params` and
        by the derivative of :meth:`torch_base`, which take the minimum-norm derivative there.
        `eigh` resolves an eigenvalue of an E with ‖E‖ ≤ 1 to about 1e-16, so the default of
        1e-10 treats anything zero up to roundoff as zero while keeping the relative error of
        each computed sqrt(λ) near 1e-6 or below.
    """

    EIGTOL_WARNING = _ot.EFFECT_EIGVAL_ABSTOL_WARN
    EIGTOL_ERROR   = _ot.EFFECT_EIGVAL_ABSTOL_ERROR
    DERIV_EIGTOL   = 1e-10

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
        """
        The element-wise derivative of this operator's superoperator, shape `(dim**2, num_params)`.

        With R = E½ the superoperator is R ⊗ Rᵀ in the standard basis, so its derivative is
        dR ⊗ Rᵀ + R ⊗ dRᵀ, where dR solves R dR + dR R = dE (see
        :func:`~pygsti.tools.matrixtools.sqrtm_frechet_derivative`). This is exact for positive
        definite E, including E with repeated eigenvalues.

        When E is singular, dR is the minimum-norm solution of that equation: in the eigenbasis
        of E, its block on the eigenvalues at or below :attr:`DERIV_EIGTOL` is zero. That is
        exact wherever E keeps its rank to first order, e.g. along a rotation of E. Along
        E(a) = P + a²D it gives 0 at a = 0, where E(a)½ has a term proportional to |a|: 0 is
        the average of the two one-sided derivatives, and what a central difference gives. So Lindblad coordinates at
        zero noise get zero columns. A parameter that moves a zero eigenvalue of E at first
        order has an infinite derivative; there this method returns the least-squares value,
        by convention and without a warning.
        """
        if self.num_params == 0:
            return LinearOperator.deriv_wrt_params(self, wrt_filter)

        # Effect parameters, placed at this operator's local indices.
        local_inds = _np.arange(self.num_params)[self._submember_rpindices[0]]
        effect_deriv = _np.zeros((self.dim, self.num_params))
        effect_deriv[:, local_inds] = self._effect.deriv_wrt_params()
        if wrt_filter is not None:
            effect_deriv = effect_deriv[:, wrt_filter]

        effect_mat = _bt.vec_to_stdmx(self._effect.to_dense(), self._basis, keep_complex=True)
        evecs, evals, _ = _mt.eigendecomposition(effect_mat, assume_hermitian=True)
        evals = _np.clip(evals, 0.0, 1.0)
        tol = self.DERIV_EIGTOL
        num_cols = effect_deriv.shape[1]

        # dE for each parameter, then dR = D sqrt(E)[dE].
        elements = _np.asarray(self._basis.elements)
        dE = _np.einsum('ik,iab->kab', effect_deriv, elements)
        dR = _mt.sqrtm_frechet_derivative(evecs, evals, dE, tol)
        R = (evecs * _np.sqrt(evals)[None, :]) @ evecs.conj().T

        # d(R ⊗ Rᵀ) = dR ⊗ Rᵀ + R ⊗ dRᵀ, as plain (non-conjugate) transposes to match rootconj_superop.
        n = R.shape[0]
        dsuperop_std = (_np.einsum('pik,jl->pijkl', dR, R.T)
                        + _np.einsum('ik,plj->pijkl', R, dR)).reshape(num_cols, n * n, n * n)
        to_basis = _BuiltinBasis('std', self.dim).create_transform_matrix(self._basis)
        from_basis = self._basis.create_transform_matrix('std')
        dsuperop = (to_basis @ dsuperop_std @ from_basis).real
        return dsuperop.reshape(num_cols, self.dim**2).T

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
        """Bake the constants used by :meth:`torch_base` (see there for the derivative convention).

        All free parameters belong to the wrapped effect; the basis element matrices ``B`` and the
        std<->basis change-of-basis matrices ``toMx`` / ``fromMx`` are constant and are pre-computed
        here (mirroring :func:`pygsti.tools.optools.rootconj_superop`), along with
        :attr:`DERIV_EIGTOL`.
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
        return (type(self._effect), self._effect.stateless_data(real_dtype, device), B, toMx, fromMx,
                self.DERIV_EIGTOL)

    @staticmethod
    def torch_base(sd: Tuple[Any, ...], t_param: _torch.Tensor) -> _torch.Tensor:
        """Differentiable reconstruction of the ρ ↦ E½ ρ E½ superoperator (see :func:`rootconj_superop`).

        .. warning::

            The derivative, in either mode (``torch.func.jacrev`` or ``jacfwd``), is exact for
            positive definite effects, including those with repeated eigenvalues. When some of
            the effect's eigenvalues are within :attr:`DERIV_EIGTOL` of 0, as for ideal
            projective effects, it is the minimum-norm derivative described in
            :meth:`deriv_wrt_params`, which agrees with that method. A parameter that moves a
            zero eigenvalue at first order has an infinite true derivative and silently gets
            the least-squares value.

        Unlike the numpy reference :func:`~pygsti.tools.optools.rootconj_superop` -- which warns
        when an eigenvalue of ``E`` strays outside ``[0, 1]`` by more than :attr:`EIGTOL_WARNING`
        and raises a ``ValueError`` past :attr:`EIGTOL_ERROR` -- this path clamps silently at any
        magnitude of violation.  So under torch-driven optimization it can keep returning
        (clamped) values in regions where the numpy path would refuse to evaluate.
        """
        etype, esd, B, toMx, fromMx, deriv_eigtol = sd
        v = etype.torch_base(esd, t_param).to(B.dtype)  # whole t_param: params delegate to the effect
        E = _torch.einsum('i,iab->ab', v, B)            # vec_to_stdmx
        E = (E + E.mH) / 2                               # match eigendecomposition's Hermitian symmetrize
        R = _EffectSqrt.apply(E, deriv_eigtol)[0]
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
