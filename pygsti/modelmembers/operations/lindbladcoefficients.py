import numpy as _np
from pygsti.tools._static_tls import import_torch as _import_torch
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
from scipy.sparse import csgraph as _csgraph
import collections as _collections
import copy as _copy
import warnings as _warnings
from pygsti.tools import lindbladtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools.graphs import sparsechol as _sparsechol
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.modelmembers import term as _term
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from functools import lru_cache, partial
try:
    from pygsti.tools import fastcalc as _fc
    triu_indices = _fc.fast_triu_indices
except ImportError:
    msg = 'Could not import cython module `fastcalc`. This may indicate that your cython extensions for pyGSTi failed to.'\
          +'properly build. Lack of cython extensions can result in significant performance degredation so we recommend trying to rebuild them.'\
           'Falling back to numpy implementation for triu_indices.'
    _warnings.warn(msg)
    triu_indices = _np.triu_indices

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero
from typing import Union, Literal
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL


def _custom_superops_stdbasis_conversion(mx_basis, sparse, superops):
    """ 
    The input superops is either:
        (1) a list of CSR matrices, or 
        (2) an ndarray of shape (n, d, d) for some d,
    where superops[i] is a superoperator in the standard (matrix-unit) basis.

    The output superops has the same datatype as the input, although
    the superoperators are now in mx_basis.
    """
    builtin_std = _BuiltinBasis("std", mx_basis.dim, sparse)
    # ^ use instead of just "std" in case mx_basis is a TensorProdBasis
    mxbasis_to_std = mx_basis.create_transform_matrix(builtin_std)

    if sparse:
        # Note: complex OK here sometimes, as only linear combos of "other" gens
        # (like (i,j) + (j,i) terms) need to be real.
        if _sps.issparse(mxbasis_to_std):
            std_to_mxbasis = _spsl.inv(mxbasis_to_std.tocsc()).tocsr()
        else:
            std_to_mxbasis = mx_basis.reverse_transform_matrix(builtin_std)
        superops = [std_to_mxbasis @ (mx @ mxbasis_to_std) for mx in superops]
        for mx in superops:
            mx.sort_indices()
    else:
        std_to_mxbasis = mx_basis.reverse_transform_matrix(builtin_std)
        # superops = _np.einsum("ik,akl,lj->aij", std_to_mxbasis, superops, mxbasis_to_std)
        temp     = _np.tensordot(std_to_mxbasis, superops, (1, 1)) # type: ignore
        temp     = _np.tensordot(temp,     mxbasis_to_std, (2, 0))
        superops = _np.transpose(temp, (1, 0, 2))
    return superops


def _unflatten_cached_other_lindblad_term_superoperators(cached_superops: Union[list, _np.ndarray], cached_superops_1norms: _np.ndarray, include_1norms: bool):
        # flat == False and block type is 'other', so need to reshape:
        nMxs = round(cached_superops_1norms.size ** 0.5)
        if isinstance(cached_superops, list):
            superops = [ [cached_superops[i * nMxs + j] for j in range(nMxs)] for i in range(nMxs) ]
        else:
            d2 = round(_np.sqrt(cached_superops.size / nMxs**2))
            superops = cached_superops.copy().reshape((nMxs, nMxs, d2, d2))
            
        if include_1norms:
            return superops, cached_superops_1norms.copy().reshape((nMxs, nMxs))
        else:
            return superops


class InvalidBlockTypeError(ValueError):

    msg = "Internal error: invalid block type!"

    def __init__(self) -> None:
        super().__init__(self.msg)


class InvalidParamModeError(ValueError):

    msg_template = "Internal error: invalid parameter mode (%s) for block type %s!"

    def __init__(self, *args: object) -> None:
        msg = InvalidParamModeError.msg_template % (args[0], args[1])
        super().__init__(msg)


# =====================================================================================================
# Parameterization strategies
#
# A `_BlockParameterization` encapsulates the map  phi : (real parameter vector v) -> block_data  for a
# coefficient block, together with its inverse and its Jacobian d(block_data)/dv.  Each block-type
# CoefficientBlock subclass exposes the parameterizations valid for it via a `_PARAM_CLASSES` dict
# {param_mode: parameterization class}; the block holds one instance in `self._parameterization` and
# delegates all param_mode-dependent numeric work to it.
#
# The *only* primitive needed to make a block Torchable is the parameterization's differentiable
# `block_data_torch` (see LindbladCoefficientBlock.torch_base), and the numpy `block_data_jacobian`
# is what the standard MapForwardSimulator path needs.
# 
#   --> Adding a new parameterization = adding one _BlockParameterization subclass and registering it.
#
# Methods take the owning block `blk` so they can read/write blk.block_data and read blk._bel_labels /
# blk._cache_mx / blk._coeff_shape.
# =====================================================================================================

class _BlockParameterization:
    """Base class for parameterization strategies; see module section header."""
    name = None  # the param_mode string this corresponds to

    def num_params(self, blk):
        raise NotImplementedError

    def params_to_block_data(self, blk, v):
        """Set blk.block_data from the parameter vector v (inverse of block_data_to_params)."""
        raise NotImplementedError

    def block_data_to_params(self, blk, ttol):
        """Return the parameter vector implied by blk.block_data (ttol = truncation tolerance)."""
        raise NotImplementedError

    def block_data_jacobian(self, blk, v):
        """Return d(block_data)/dv with shape blk._coeff_shape + (num_params,)."""
        raise NotImplementedError

    def param_labels(self, blk):
        raise NotImplementedError

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        """Second derivative of the term-superoperators w.r.t. params (rarely used)."""
        raise NotImplementedError

    # --- Torchable support (differentiable params -> block_data) ---
    def torch_stateless_data(self, blk, real_dtype, device):
        """Constants (independent of the free params) needed by `block_data_torch`."""
        return (len(blk._bel_labels),)

    @staticmethod
    def block_data_torch(sd, t_param):
        """Differentiable torch reconstruction of block_data from the parameter tensor t_param,
        given the stateless data `sd` returned by `torch_stateless_data`."""
        raise NotImplementedError

    def _coefficient_polynomial(self, blk, index: int | tuple[int, int], pio: int, mpv: int) -> _Polynomial:
        """Return the _Polynomial for one block_data entry.

        Evaluating the returned polynomial at the block's parameter vector
        (placed at offset ``pio``) must equal the corresponding value produced
        by ``params_to_block_data`` (i.e. ``blk.block_data[index]``).

        Parameters
        ----------
        blk : LindbladCoefficientBlock
            The owning coefficient block.
        index : int or tuple of ints
            Integer coefficient index for vector blocks, or an (i, j) tuple
            for matrix blocks.
        pio : int
            Global parameter index offset for polynomial variables.
        mpv : int
            Maximum number of polynomial variables.

        Returns
        -------
        Polynomial
        """
        raise NotImplementedError



class _StaticParam(_BlockParameterization):
    """Zero-parameter block: block_data is a fixed constant."""
    name = 'static'

    def num_params(self, blk):
        return 0

    def params_to_block_data(self, blk, v):
        assert len(v) == 0, "'static' parameterized blocks should have zero parameters!"

    def block_data_to_params(self, blk, ttol):
        return _np.empty(0, 'd')

    def block_data_jacobian(self, blk, v):
        return _np.zeros(blk._coeff_shape + (0,), 'd')

    def param_labels(self, blk):
        return []

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        if superops_are_flat or not blk._superops_matrix_structured:
            return _np.zeros((superops.shape[1], superops.shape[2], 0, 0), 'd')
        return _np.zeros((superops.shape[2], superops.shape[3], 0, 0), 'd')

    def torch_stateless_data(self, blk, real_dtype, device):
        _torch = _import_torch()
        arr = _np.ascontiguousarray(blk.block_data)
        ten = _torch.from_numpy(arr).to(dtype=real_dtype, device=device)
        return (ten,)

    @staticmethod
    def block_data_torch(sd, t_param):
        return sd[0]

    def _coefficient_polynomial(self, blk, index: int | tuple[int, int], pio: int, mpv: int) -> _Polynomial:
        val = blk.block_data[index]
        return _Polynomial({(): val}, mpv)


class _RealVectorElements(_BlockParameterization):
    """Unconstrained real-vector parameterization: block_data == v (identity map).  Shared numeric
    behavior for the 'ham' and 'other_diagonal' 'elements' modes; subclasses set the labels."""
    name = 'elements'

    def num_params(self, blk):
        return len(blk._bel_labels)

    def params_to_block_data(self, blk, v):
        blk.block_data[:] = v

    def block_data_to_params(self, blk, ttol):
        return blk.block_data.real

    def block_data_jacobian(self, blk, v):
        return _np.identity(len(blk._bel_labels), 'd')

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        d1, d2 = superops.shape[1:3]
        nP = blk.num_params
        return _np.zeros((d1, d2, nP, nP), 'd')

    @staticmethod
    def block_data_torch(sd, t_param):
        return t_param

    def _coefficient_polynomial(self, blk, index: int, pio: int, mpv: int) -> _Polynomial:
        return _Polynomial({(pio + index,): 1.0}, mpv)


class _HamElements(_RealVectorElements):
    def param_labels(self, blk):
        return [f"{lbl} Hamiltonian error coefficient" for lbl in blk._bel_labels]


class _DiagElements(_RealVectorElements):
    def param_labels(self, blk):
        return [f"{lbl} stochastic coefficient" for lbl in blk._bel_labels]


class _EEGVectorElements(_RealVectorElements):
    """Unconstrained real-vector parameterization for blocks whose coefficients are in one-to-one
    correspondence with an explicit list of elementary error generators (block_data == v).  Like
    ``_RealVectorElements`` but sized by ``blk._eeg_labels`` rather than ``blk._bel_labels``."""
    def num_params(self, blk):
        return len(blk._eeg_labels)

    def block_data_jacobian(self, blk, v):
        return _np.identity(len(blk._eeg_labels), 'd')

    def torch_stateless_data(self, blk, real_dtype, device):
        # Sized by _eeg_labels (not _bel_labels): these blocks have one parameter per elementary
        # error generator.  The identity block_data_torch is inherited from _RealVectorElements.
        return (len(blk._eeg_labels),)

    def param_labels(self, blk):
        labels = []
        for lbl in blk._eeg_labels:            
            if lbl.errorgen_type == 'S':
                labels.append(f"{lbl} stochastic coefficient")
            elif lbl.errorgen_type == 'C':
                labels.append(f"{lbl} C-type correlation coefficient")
            else: #'A'
                labels.append(f"{lbl} A-type active coefficient")
        return labels


class _DiagCholesky(_BlockParameterization):
    """Diagonal Pauli-stochastic CPTP parameterization: block_data[i] = v[i]**2 (>= 0)."""
    name = 'cholesky'

    def num_params(self, blk):
        return len(blk._bel_labels)

    def params_to_block_data(self, blk, v):
        blk.block_data[:] = v**2

    def block_data_to_params(self, blk, ttol):
        nonneg = [val >= ttol for val in blk.block_data]
        assert all(nonneg), "Lindblad stochastic coefficients are not positive!"
        clipped = blk.block_data.clip(0, 1e100)
        return _np.sqrt(clipped.real)

    def block_data_jacobian(self, blk, v):
        num_bels = len(blk._bel_labels)
        assert len(v) == num_bels, f"Expected {num_bels} parameters, got {len(v)}!"
        # block_data[i] = v[i]**2  =>  d(block_data[i])/d v[j] = 2 v[i] delta_{i,j}
        return 2.0 * _np.diag(v)

    def param_labels(self, blk):
        return [f"sqrt({lbl} stochastic coefficient)" for lbl in blk._bel_labels]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        nP = blk.num_params
        trans = _np.transpose(superops, (1, 2, 0))      # shape (d,d,nP)
        I = _np.identity(nP, 'd')
        return trans[:, :, :, None] * (2.0 * I)[None, None, :, :]

    @staticmethod
    def block_data_torch(sd, t_param):
        return t_param**2

    def _coefficient_polynomial(self, blk, index: int, pio: int, mpv: int) -> _Polynomial:
        var = pio + index
        return _Polynomial({(var, var): 1.0}, mpv)


class _Depol(_BlockParameterization):
    """Depolarizing: a single nonnegative coefficient shared by all diagonal generators
    (block_data[i] = v[0]**2 for all i)."""
    name = 'depol'

    def num_params(self, blk):
        return 1

    def params_to_block_data(self, blk, v):
        blk.block_data[:] = (v.item()**2) * _np.ones(len(blk._bel_labels), 'd')

    def block_data_to_params(self, blk, ttol):
        nonneg = [val >= ttol for val in blk.block_data]
        assert all(nonneg), f"Lindblad stochastic coefficients are not positive! (tol={ttol})"
        first = blk.block_data[0]
        all_equal = _np.isclose(blk.block_data, first, atol=1e-6)
        assert all(all_equal), f"Diagonal lindblad coefficients are not equal! (tol={ttol})"
        avg = _np.mean(blk.block_data.clip(0, 1e100))
        return _np.array([_np.sqrt(_np.real(avg))], 'd')

    def block_data_jacobian(self, blk, v):
        mat = _np.zeros((len(blk._bel_labels), 1), 'd')
        mat[:, 0] = 2.0 * v[0]
        return mat

    def param_labels(self, blk):
        return ["sqrt(common stochastic error coefficient for depolarization)"]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        base = _np.sum(superops, axis=0)   # shape (d,d)
        return base[:, :, None, None] * 2.0

    @staticmethod
    def block_data_torch(sd, t_param):
        _torch = _import_torch()
        n = sd[0]
        return (t_param[0]**2) * _torch.ones(n, dtype=t_param.dtype, device=t_param.device)

    def _coefficient_polynomial(self, blk, index: int, pio: int, mpv: int) -> _Polynomial:
        var = pio
        return _Polynomial({(var, var): 1.0}, mpv)


class _RelDepol(_BlockParameterization):
    """Relative depolarizing: a single (possibly negative) coefficient shared by all diagonal
    generators (block_data[i] = v[0] for all i)."""
    name = 'reldepol'

    def num_params(self, blk):
        return 1

    def params_to_block_data(self, blk, v):
        blk.block_data[:] = v.item() * _np.ones(len(blk._bel_labels), 'd')

    def block_data_to_params(self, blk, ttol):
        first = blk.block_data[0]
        all_equal = [_np.isclose(val, first, atol=1e-6) for val in blk.block_data]
        assert all(all_equal), f"Diagonal lindblad coefficients are not equal! (tol={ttol})"
        avg = _np.mean(blk.block_data)
        return _np.array([_np.real(avg)], 'd')

    def block_data_jacobian(self, blk, v):
        mat = _np.zeros((len(blk._bel_labels), 1), 'd')
        mat[:, 0] = 1.0
        return mat

    def param_labels(self, blk):
        return ["common stochastic error coefficient for depolarization"]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        d1, d2 = superops.shape[1:3]
        return _np.zeros((d1, d2, 1, 1), 'd')

    @staticmethod
    def block_data_torch(sd, t_param):
        _torch = _import_torch()
        n = sd[0]
        return t_param[0] * _torch.ones(n, dtype=t_param.dtype, device=t_param.device)

    def _coefficient_polynomial(self, blk, index: int, pio: int, mpv: int) -> _Polynomial:
        return _Polynomial({(pio,): 1.0}, mpv)


class _OtherElements(_BlockParameterization):
    """Unconstrained Hermitian-matrix parameterization for the full 'other' block.  The n*n real
    params encode the Hermitian block_data: diag = real diagonal, lower = real parts, upper = imag parts."""
    name = 'elements'

    def num_params(self, blk):
        return len(blk._bel_labels)**2

    def params_to_block_data(self, blk, v):
        # Maps real parameter vector v to complex Hermitian block_data.
        # Parameter matrix P = reshape(v, (n, n)):
        #   diag:  block_data[i, i] = P[i, i]
        #   lower: block_data[i, j] = P[i, j] + 1j * P[j, i]  (i > j)
        #   upper: block_data[i, j] = P[j, i] - 1j * P[i, j]  (i < j, by Hermiticity)
        # Keep conventions in sync with _coefficient_polynomial().
        num_bels = len(blk._bel_labels)
        params = v.reshape((num_bels, num_bels))
        params_upper_indices = triu_indices(num_bels)
        params_upper = -1j * params[params_upper_indices]
        params_lower = (params.T)[params_upper_indices]
        block_data_trans = blk.block_data.T
        blk.block_data[params_upper_indices] = params_lower + params_upper
        block_data_trans[params_upper_indices] = params_lower - params_upper
        diag_indices = cached_diag_indices(num_bels)
        blk.block_data[diag_indices] = params[diag_indices]

    def block_data_to_params(self, blk, ttol):
        is_hermitian = _mt.is_hermitian(blk.block_data, 1e-8)
        assert is_hermitian, "Lindblad 'other' coefficient matrix is not Hermitian!"
        num_bels = len(blk._bel_labels)
        params_mat = _np.empty((num_bels, num_bels), 'd')
        for i in range(num_bels):
            assert _np.abs(blk.block_data[i, i].imag) < IMAG_TOL
            params_mat[i, i] = blk.block_data[i, i].real
            for j in range(i):
                params_mat[i, j] = blk.block_data[i, j].real
                params_mat[j, i] = blk.block_data[i, j].imag
        return params_mat.ravel()

    def block_data_jacobian(self, blk, v):
        num_bels = len(blk._bel_labels)
        nP = len(v)
        stride = num_bels
        mat = _np.zeros((num_bels, num_bels, nP), 'complex')
        for i in range(num_bels):
            mat[i, i, i * stride + i] = 1.0
            for j in range(i):
                mat[i, j, i * stride + j] = 1.0
                mat[i, j, j * stride + i] = 1.0j
                mat[j, i, i * stride + j] = 1.0
                mat[j, i, j * stride + i] = -1.0j
        return mat

    def param_labels(self, blk):
        labels = []
        for i, ilbl in enumerate(blk._bel_labels):
            for j, jlbl in enumerate(blk._bel_labels):
                if i == j:
                    labels.append(f"{ilbl} diagonal element of non-H coeff matrix")
                elif j < i:
                    labels.append(f"Re[({ilbl},{jlbl}) element of non-H coeff matrix]")
                else:
                    labels.append(f"Im[({ilbl},{jlbl}) element of non-H coeff matrix]")
        return labels

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        num_bels = len(blk._bel_labels)
        if superops_are_flat:
            superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
        snP = num_bels
        # unconstrained Hermitian: Hessian is identically zero
        return _np.zeros([superops.shape[2], superops.shape[3], snP, snP, snP, snP], 'd')

    @staticmethod
    def block_data_torch(sd, t_param):
        _torch = _import_torch()
        n = sd[0]
        P = t_param.reshape(n, n)
        lowP = _torch.tril(P, -1)
        Re = lowP + lowP.T + _torch.diag(_torch.diagonal(P))
        lowPT = _torch.tril(P.T, -1)
        Imag = lowPT - lowPT.T
        return _torch.complex(Re, Imag)

    def _coefficient_polynomial(self, blk, index: tuple[int, int], pio: int, mpv: int) -> _Polynomial:
        # Parameter matrix P = reshape(v, (n, n)) encodes Hermitian block_data:
        #   diag:  block_data[i, i] = P[i, i] (real)
        #   lower: block_data[i, j] = P[i, j] + 1j * P[j, i]  (i > j)
        #   upper: block_data[i, j] = P[j, i] - 1j * P[i, j]  (i < j, by Hermiticity)
        # Keep conventions in sync with params_to_block_data().
        i, j = index
        num_bels = len(blk._bel_labels)
        if i == j:
            var_ii = pio + (i * num_bels + i)
            return _Polynomial({(var_ii,): 1.0}, mpv)
        elif i > j:
            var_ij = pio + (i * num_bels + j)
            var_ji = pio + (j * num_bels + i)
            return _Polynomial({(var_ij,): 1.0, (var_ji,): 1.0j}, mpv)
        else:  # i < j
            var_ji = pio + (j * num_bels + i)
            var_ij = pio + (i * num_bels + j)
            return _Polynomial({(var_ji,): 1.0, (var_ij,): -1.0j}, mpv)


class _OtherCholesky(_BlockParameterization):
    """CPTP (positive-semidefinite) parameterization for the full 'other' block: block_data = C C^dag,
    where the n*n real params encode the lower-triangular Cholesky factor C."""
    name = 'cholesky'

    def num_params(self, blk):
        return len(blk._bel_labels)**2

    def params_to_block_data(self, blk, v):
        # Maps real parameter vector v to positive semidefinite block_data = C @ C^dag,
        # where C is lower-triangular with real diagonal elements.
        # Keep conventions in sync with _coefficient_polynomial().
        num_bels = len(blk._bel_labels)
        params = v.reshape((num_bels, num_bels))
        params_upper_indices = triu_indices(num_bels)
        cache_mx = blk._cache_mx
        params_upper = 1j * params[params_upper_indices]
        params_lower = (params.T)[params_upper_indices]
        cache_mx_trans = cache_mx.T
        cache_mx_trans[params_upper_indices] = params_lower + params_upper
        diag_indices = cached_diag_indices(num_bels)
        cache_mx[diag_indices] = params[diag_indices]
        blk.block_data[:, :] = cache_mx @ cache_mx.T.conj()

    def block_data_to_params(self, blk, ttol):
        is_hermitian = _mt.is_hermitian(blk.block_data, 1e-8)
        assert is_hermitian, "Lindblad 'other' coefficient matrix is not Hermitian!"
        num_bels = len(blk._bel_labels)
        params_mat = _np.empty((num_bels, num_bels), 'd')

        col_norms = _np.linalg.norm(blk.block_data, axis=0)
        row_norms = _np.linalg.norm(blk.block_data, axis=1)
        ind_weights = col_norms + row_norms
        zero_inds = set(_np.where(ind_weights < 1e-12 * num_bels)[0])
        num_nonzero = num_bels - len(zero_inds)

        next_nonzero = 0; next_zero = num_nonzero
        perm = _np.zeros((num_bels, num_bels), 'd')
        for i in range(num_bels):
            if i in zero_inds:
                perm[next_zero, i] = 1.0; next_zero += 1
            else:
                perm[next_nonzero, i] = 1.0; next_nonzero += 1

        pdata = perm @ blk.block_data @ perm.T
        small = pdata[0:num_nonzero, 0:num_nonzero]
        evals, U = _np.linalg.eigh(small)
        assert all([ev > ttol for ev in evals]), "Lindblad coefficients are not CPTP!"
        if ttol < 0:
            Ui = U.T.conj()
            pev = evals.clip(1e-16, None)
            small = U @ (pev[:, None] * Ui)
            try:
                Lsmall = _np.linalg.cholesky(small)
            except _np.linalg.LinAlgError:
                pev = evals.clip(1e-12, 1e100)
                small = U @ (pev[:, None] * Ui)
                Lsmall = _np.linalg.cholesky(small)
        else:
            Lsmall = _np.linalg.cholesky(small)

        perm_Lmx = _np.zeros((num_bels, num_bels), 'complex')
        perm_Lmx[:num_nonzero, :num_nonzero] = Lsmall
        Lmx = perm.T @ perm_Lmx @ perm

        for i in range(num_bels):
            assert _np.abs(Lmx[i, i].imag) < IMAG_TOL
            params_mat[i, i] = Lmx[i, i].real
            for j in range(i):
                params_mat[i, j] = Lmx[i, j].real
                params_mat[j, i] = Lmx[i, j].imag
        return params_mat.ravel()

    def block_data_jacobian(self, blk, v):
        num_bels = len(blk._bel_labels)
        nP = len(v)
        params = v.reshape((num_bels, num_bels))
        stride = num_bels
        dC = _np.zeros((nP, num_bels, num_bels), 'complex')
        C = blk._cache_mx
        for i in range(num_bels):
            C[i, i] = params[i, i]
            dC[i * stride + i, i, i] = 1.0
            for j in range(i):
                C[i, j] = params[i, j] + 1j * params[j, i]
                dC[i * stride + j, i, j] = 1.0
                dC[j * stride + i, i, j] = 1.0j
        dBdp = _np.dot(dC, C.T.conjugate())
        dBdp += _np.dot(C, dC.conjugate().transpose((0, 2, 1))).transpose((1, 0, 2))
        dBdp = _np.rollaxis(dBdp, 0, 3)  # => (num_bels, num_bels, nP)
        return dBdp

    def param_labels(self, blk):
        labels = []
        for i, ilbl in enumerate(blk._bel_labels):
            for j, jlbl in enumerate(blk._bel_labels):
                if i == j:
                    labels.append(f"{ilbl} diagonal element of non-H coeff Cholesky decomp")
                elif j < i:
                    labels.append(f"Re[({ilbl},{jlbl}) element of non-H coeff Cholesky decomp]")
                else:
                    labels.append(f"Im[({ilbl},{jlbl}) element of non-H coeff Cholesky decomp]")
        return labels

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        num_bels = len(blk._bel_labels)
        if superops_are_flat:
            superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
        snP = num_bels
        d2Odp2 = _np.zeros([superops.shape[2], superops.shape[3], snP, snP, snP, snP], 'complex')

        def iter_base_ab_qr(ab_inc_eq, qr_inc_eq):
            for _base in range(snP):
                start_ab = _base if ab_inc_eq else _base + 1
                start_qr = _base if qr_inc_eq else _base + 1
                for _ab in range(start_ab, snP):
                    for _qr in range(start_qr, snP):
                        yield (_base, _ab, _qr)

        for base, a, q in iter_base_ab_qr(True, True):
            d2Odp2[:, :, a, base, q, base] = superops[a, q] + superops[q, a]
        for base, a, r in iter_base_ab_qr(True, False):
            d2Odp2[:, :, a, base, base, r] = -1j * superops[a, r] + 1j * superops[r, a]
        for base, b, q in iter_base_ab_qr(False, True):
            d2Odp2[:, :, base, b, q, base] = 1j * superops[b, q] - 1j * superops[q, b]
        for base, b, r in iter_base_ab_qr(False, False):
            d2Odp2[:, :, base, b, base, r] = superops[b, r] + superops[r, b]
        return d2Odp2

    @staticmethod
    def block_data_torch(sd, t_param):
        _torch = _import_torch()
        n = sd[0]
        P = t_param.reshape(n, n)
        Cre = _torch.tril(P, -1) + _torch.diag(_torch.diagonal(P))
        Cim = _torch.tril(P.T, -1)
        C = _torch.complex(Cre, Cim)
        return C @ C.conj().T

    def _coefficient_polynomial(self, blk, index: tuple[int, int], pio: int, mpv: int) -> _Polynomial:
        # block_data = C @ C^dag where lower-triangular Cholesky factor C has:
        #   Re(C[a, b]) = P[a, b]  for b <= a
        #   Im(C[a, b]) = P[b, a]  for b < a  (and 0 for b == a)
        #
        # block_data[i, j] = sum_k C[i, k] * conj(C[j, k])
        #                  = sum_k (Re(C[i, k]) + 1j*Im(C[i, k])) * (Re(C[j, k]) - 1j*Im(C[j, k]))
        # Keep conventions in sync with params_to_block_data().
        i, j = index
        num_bels = len(blk._bel_labels)

        def i_re(a, b): return pio + (a * num_bels + b)
        def i_im(a, b): return pio + (b * num_bels + a)

        polyTerms = {}
        for k in range(0, min(i, j) + 1):
            if k <= i and k <= j:
                polyTerms[(i_re(i, k), i_re(j, k))] = 1.0
            if k <= i and k < j:
                polyTerms[(i_re(i, k), i_im(j, k))] = -1.0j
            if k < i and k <= j:
                polyTerms[(i_im(i, k), i_re(j, k))] = 1.0j
            if k < i and k < j:
                polyTerms[(i_im(i, k), i_im(j, k))] = 1.0
        return _Polynomial(polyTerms, mpv)


class _SparseCholesky(_BlockParameterization):
    """Exact quadratic map from a fill-free Cholesky factor to selected S/C/A rates."""
    name = 'cholesky'

    def num_params(self, blk):
        return len(blk._factor_rows)

    def params_to_block_data(self, blk, v):
        if _np.shape(v) != (blk.num_params,):
            raise ValueError("Expected %d sparse Cholesky parameters." % blk.num_params)
        terms, left, right, scale = blk._quadratic_terms
        blk.block_data[:] = _np.bincount(terms, weights=scale * v[left] * v[right],
                                        minlength=len(blk._eeg_labels))

    def block_data_to_params(self, blk, ttol):
        K = blk.kossakowski_matrix()
        n = len(K)
        if n == 0:
            return _np.empty(0, 'd')
        # A diagonal shift, unlike spectral clipping, preserves the prescribed zeros.
        emin = _np.linalg.eigvalsh(K)[0]
        if emin < ttol:
            raise ValueError("Sparse Lindblad coefficients are not positive semidefinite.")
        if emin < 0:
            K = K - emin * _np.eye(n)
        K = K[_np.ix_(blk.cholesky_structure.perm, blk.cholesky_structure.perm)]
        L = _np.zeros_like(K)
        tol = 1e-12 * max(_np.linalg.norm(K, ord=_np.inf), _np.finfo(float).tiny)
        for j in range(n):
            pivot = (K[j, j] - L[j, :j] @ L[j, :j].conj()).real
            residual = K[j + 1:, j] - L[j + 1:, :j] @ L[j, :j].conj()
            if pivot < -tol:
                raise ValueError("Sparse Lindblad coefficients are not positive semidefinite.")
            if pivot <= tol and _np.all(_np.abs(residual) <= tol):
                continue  # A zero pivot in a PSD matrix has a zero residual column.
            if pivot <= 0:
                raise ValueError("Cannot factor sparse Lindblad coefficients at a zero pivot.")
            L[j, j] = _np.sqrt(pivot)
            L[j + 1:, j] = residual / L[j, j]
        entries = L[blk._factor_rows, blk._factor_cols]
        v = _np.where(blk._factor_imag, entries.imag, entries.real)
        # Check the sparse factor, including singular inputs, before returning parameters.
        sparse_L = _np.zeros_like(L)
        _np.add.at(sparse_L, (blk._factor_rows, blk._factor_cols), v * blk._factor_scales)
        if _np.linalg.norm(sparse_L @ sparse_L.conj().T - K, ord=_np.inf) > 100 * tol:
            raise ValueError("Sparse Cholesky reconstruction failed for these coefficients.")
        return v

    def block_data_jacobian(self, blk, v):
        terms, left, right, scale = blk._quadratic_terms
        J = _np.zeros((len(blk._eeg_labels), len(v)))
        _np.add.at(J, (terms, left), scale * v[right])
        _np.add.at(J, (terms, right), scale * v[left])
        return J

    def param_labels(self, blk):
        perm = blk.cholesky_structure.perm
        return ["%s[(%s,%s) sparse non-H Cholesky element]" %
                ('Im' if imag else 'Re', blk._bel_labels[perm[r]], blk._bel_labels[perm[c]])
                for r, c, imag in zip(blk._factor_rows, blk._factor_cols, blk._factor_imag)]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        nP = blk.num_params
        out = _np.zeros(superops.shape[1:] + (nP, nP), dtype=superops.dtype)
        terms, left, right, scale = blk._quadratic_terms
        for t, p, q, s in zip(terms, left, right, scale):
            out[:, :, p, q] += s * superops[t]
            out[:, :, q, p] += s * superops[t]
        return out

    def torch_stateless_data(self, blk, real_dtype, device):
        torch = _import_torch()
        terms, left, right, scale = blk._quadratic_terms
        return (len(blk._eeg_labels),
                *(torch.tensor(a, dtype=torch.long, device=device) for a in (terms, left, right)),
                torch.tensor(scale, dtype=real_dtype, device=device))

    @staticmethod
    def block_data_torch(sd, t_param):
        n, terms, left, right, scale = sd
        return t_param.new_zeros(n).scatter_add(0, terms, scale * t_param[left] * t_param[right])

    def _coefficient_polynomial(self, blk, index, pio, mpv):
        terms, left, right, scale = blk._quadratic_terms
        selected = terms == index
        return _Polynomial({(pio + int(p), pio + int(q)): s
                            for p, q, s in zip(left[selected], right[selected], scale[selected])}, mpv)


class _CompletedCholesky(_BlockParameterization):
    """Interior sparse CPTP chart with dependent factor components.

    Write L = D T with positive diagonal D and unit-lower T. Parameters are log(D_ii)
    and the allowed components of T. Missing components are completed in T, without
    pivot divisions. Column normalization is an alternative chart and supports earlier artifacts.
    """
    name = 'cholesky completion'

    def num_params(self, blk):
        return blk.cholesky_structure.n + len(blk._completion_free_params)

    @staticmethod
    def _gram_jet(values, gradients, hessians, left, right):
        a, b = values[left], values[right].conj()
        value = a @ b
        gradient = hessian = None
        if gradients is not None:
            ga, gb = gradients[left], gradients[right].conj()
            gradient = a @ gb + b @ ga
        if hessians is not None:
            cross = ga.T @ gb
            hessian = (_np.einsum('m,mpq->pq', a, hessians[right].conj())
                       + _np.einsum('m,mpq->pq', b, hessians[left]) + cross + cross.T)
        return value, gradient, hessian

    def _factor_jet(self, blk, v, order=0, directions=None):
        """Factor and optional first/second derivatives along columns of ``directions``."""
        n = blk.cholesky_structure.n
        if _np.shape(v) != (blk.num_params,) or not _np.all(_np.isfinite(v)):
            raise ValueError("Expected %d finite completed-Cholesky parameters." % blk.num_params)
        with _np.errstate(over='ignore', under='ignore'):
            diagonal = _np.exp(v[:n])
        if _np.any(diagonal == 0) or not _np.all(_np.isfinite(diagonal)):
            raise ValueError("Completed-Cholesky log pivots exceed floating-point range.")
        cols = blk._completion_cols
        row_normalized = blk._completion_normalization == 'row'
        free, params, scales = blk._completion_free, blk._completion_free_params, blk._completion_scales
        values = _np.zeros(len(cols), complex)
        values[blk._completion_diagonal] = 1.
        _np.add.at(values, free, scales * v[params])
        if not row_normalized:
            values *= diagonal[cols]
        gradients = hessians = None
        if order:
            directions = _np.eye(blk.num_params) if directions is None else _np.asarray(directions)
            log_deriv = directions[cols]
            gradients = _np.zeros((len(cols), directions.shape[1]), complex) if row_normalized \
                else values[:, None] * log_deriv
            edge_scales = scales if row_normalized else scales * diagonal[cols[free]]
            _np.add.at(gradients, free, edge_scales[:, None] * directions[params])
        if order == 2:
            hessians = _np.zeros((len(cols), directions.shape[1], directions.shape[1]), complex)
            if not row_normalized:
                hessians += values[:, None, None] * log_deriv[:, :, None] * log_deriv[:, None, :]
                edge_deriv = (scales * diagonal[cols[free]])[:, None] * directions[params]
                cross = edge_deriv[:, :, None] * directions[cols[free]][:, None, :]
                _np.add.at(hessians, free, cross + cross.transpose(0, 2, 1))

        for index, col, real_free, imag_free, left, right in blk._completion_steps:
            s, gs, hs = self._gram_jet(values, gradients, hessians, left, right)
            divisor = 1. if row_normalized else diagonal[col]
            q = -s / divisor
            values[index] = ((values[index].real if real_free else q.real)
                             + 1j * (values[index].imag if imag_free else q.imag))
            if order:
                d = _np.zeros(directions.shape[1]) if row_normalized else directions[col]
                gq = (-gs + s * d) / divisor
                gradients[index] = ((gradients[index].real if real_free else gq.real)
                                     + 1j * (gradients[index].imag if imag_free else gq.imag))
            if order == 2:
                hq = (-hs + _np.outer(gs, d) + _np.outer(d, gs) - s * _np.outer(d, d)) / divisor
                hessians[index] = ((hessians[index].real if real_free else hq.real)
                                   + 1j * (hessians[index].imag if imag_free else hq.imag))
        if row_normalized:
            rows = blk._completion_rows
            if order == 2:
                d = directions[rows]
                cross = gradients[:, :, None] * d[:, None, :]
                hessians = diagonal[rows, None, None] * (
                    hessians + cross + cross.transpose(0, 2, 1) + values[:, None, None] * d[:, :, None] * d[:, None, :])
            if order:
                gradients = diagonal[rows, None] * (gradients + values[:, None] * directions[rows])
            values *= diagonal[rows]
        if not _np.all(_np.isfinite(values)):
            raise ValueError("Completed-Cholesky factor overflow; check pivot sizes and free coordinates.")
        return values, gradients, hessians

    def _evaluate(self, blk, v, order=0, directions=None):
        values, gradients, hessians = self._factor_jet(blk, v, order, directions)
        if not order:
            out = _np.zeros(len(blk._eeg_labels), complex)
            terms, left, right = blk._completion_gram
            _np.add.at(out, terms, values[left] * values[right].conj())
            return _np.where(blk._completion_active, -out.imag, out.real), None, None
        data, jacobian, hessian = [], [], []
        for active, (left, right) in zip(blk._completion_active, blk._completion_coefficient_terms):
            value, gradient, second = self._gram_jet(values, gradients, hessians, left, right)
            data.append(-value.imag if active else value.real)
            jacobian.append(-gradient.imag if active else gradient.real)
            if order == 2:
                hessian.append(-second.imag if active else second.real)
        return _np.array(data), _np.array(jacobian), _np.array(hessian) if order == 2 else None

    def params_to_block_data(self, blk, v):
        data = self._evaluate(blk, v)[0]
        if not _np.all(_np.isfinite(data)):
            raise ValueError("Completed-Cholesky coefficients exceed floating-point range.")
        blk.block_data[:] = data

    def block_data_to_params(self, blk, ttol):
        st = blk.cholesky_structure
        if not st.n:
            return _np.empty(0, 'd')
        K = blk.kossakowski_matrix()[_np.ix_(st.perm, st.perm)]
        if not _np.all(_np.isfinite(K)):
            raise ValueError("Completed Cholesky requires finite positive-definite coefficients.")
        # A singular PSD matrix can acquire a tiny positive last Cholesky pivot
        # through roundoff. Check numerical rank after diagonal equilibration,
        # so uniform scaling or a small physical row does not trigger rejection.
        diagonal = K.diagonal().real
        if _np.any(diagonal <= 0):
            raise ValueError("Completed Cholesky requires positive-definite coefficients; "
                             "zero noise and the singular PSD boundary are outside this chart.")
        scale = _np.sqrt(diagonal)
        equilibrated = (K / scale[:, None]) / scale[None, :]
        spectrum = _np.linalg.eigvalsh(equilibrated)
        if spectrum[0] <= st.n * _np.finfo(float).eps * spectrum[-1]:
            raise ValueError("Completed Cholesky requires numerically positive-definite coefficients; "
                             "the singular PSD boundary is outside this chart.")
        try:
            L = _np.linalg.cholesky(K)
        except _np.linalg.LinAlgError as exc:
            raise ValueError("Completed Cholesky requires positive-definite coefficients; "
                             "zero noise and the singular PSD boundary are outside this chart.") from exc
        diagonal = L.diagonal().real
        entries = L[blk._completion_rows[blk._completion_free], blk._completion_cols[blk._completion_free]]
        pivot_indices = blk._completion_rows if blk._completion_normalization == 'row' else blk._completion_cols
        normalized = entries / diagonal[pivot_indices[blk._completion_free]]
        return _np.r_[_np.log(diagonal), _np.where(blk._completion_scales.imag != 0,
                                                  normalized.imag, normalized.real)]

    def block_data_jacobian(self, blk, v):
        return self._evaluate(blk, v, order=1)[1]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        hessian = self._evaluate(blk, blk.to_vector() if v is None else v, order=2)[2]
        return _np.einsum('cpq,cij->ijpq', hessian, superops)

    def param_labels(self, blk):
        st = blk.cholesky_structure
        labels = ["log[%s completed non-H Cholesky pivot]" % blk._bel_labels[i] for i in st.perm]
        for f, scale in zip(blk._completion_free, blk._completion_scales):
            r, c = blk._completion_rows[f], blk._completion_cols[f]
            pivot = r if blk._completion_normalization == 'row' else c
            labels.append("%s[L(%s,%s)/L(%s,%s)]" %
                          ('Im' if scale.imag else 'Re', blk._bel_labels[st.perm[r]],
                           blk._bel_labels[st.perm[c]], blk._bel_labels[st.perm[pivot]], blk._bel_labels[st.perm[pivot]]))
        return labels

    def torch_stateless_data(self, blk, real_dtype, device):
        torch = _import_torch()
        index = lambda a: torch.tensor(a, dtype=torch.long, device=device)
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        steps = [(i, j, re, im, left.tolist(), right.tolist())
                 for i, j, re, im, left, right in blk._completion_steps]
        return (blk.cholesky_structure.n, index(blk._completion_cols), index(blk._completion_diagonal),
                index(blk._completion_free), index(blk._completion_free_params),
                torch.tensor(blk._completion_scales, dtype=complex_dtype, device=device), steps,
                tuple(index(a) for a in blk._completion_gram),
                torch.tensor(blk._completion_active, dtype=torch.bool, device=device),
                index(blk._completion_rows), blk._completion_normalization == 'row')

    @staticmethod
    def block_data_torch(sd, t_param):
        torch = _import_torch()
        n, cols, diag_indices, free, params, scales, steps, gram, active, rows, row_normalized = sd
        diagonal = t_param[:n].exp()
        base = scales.new_zeros(len(cols)).index_add(0, diag_indices, scales.new_ones(n))
        base = base.index_add(0, free, scales * t_param[params])
        entries = list((base if row_normalized else base * diagonal[cols]).unbind())
        for index, col, real_free, imag_free, left, right in steps:
            s = (torch.stack([entries[a] for a in left])
                 * torch.stack([entries[b].conj() for b in right])).sum() if left else base.new_zeros(())
            q = -s if row_normalized else -s / diagonal[col]
            entries[index] = torch.complex(entries[index].real if real_free else q.real,
                                            entries[index].imag if imag_free else q.imag)
        factor = torch.stack(entries) if entries else base
        if row_normalized:
            factor = factor * diagonal[rows]
        terms, left, right = gram
        result = base.new_zeros(len(active)).index_add(0, terms, factor[left] * factor[right].conj())
        return torch.where(active, -result.imag, result.real)

    def _coefficient_polynomial(self, blk, index, pio, mpv):
        raise NotImplementedError("Completed Cholesky is nonpolynomial; use matrix simulation or Torch, "
                                  "rather than polynomial term simulation.")


class _Contraction(_BlockParameterization):
    """Sparse CPTP chart K = C (I + H) C that includes zero noise and the PSD boundary.

    C is a real diagonal matrix with C_ii^2 the S coefficients, and H is zero-diagonal
    Hermitian with entries only on the selected C/A components. K is positive semidefinite
    exactly when lambda_min(H) >= -1, on each connected component of the C/A graph. The
    parameters are C_ii and a free H0 with the same support, scaled per component by
    ``H = g(t) H0``, where ``t = -lambda_min(H0)`` and ``g(t) = 2 / (1 + t^2)``. Since H0
    has zero trace, t >= 0 with equality only at H0 = 0, and ``-lambda_min(H) = 2t / (1 + t^2)``
    covers [0, 1]. A scalar scaling preserves every support, so no completion is needed.
    The boundary ``t = 1`` is a fold. Derivatives of t do not exist where lambda_min(H0)
    is repeated; there the code uses the derivative along one eigenvector.
    """
    name = 'contraction'

    def num_params(self, blk):
        return len(blk._bel_labels) + len(blk._contraction_terms[0])

    @staticmethod
    def _component_spectrum(blk, w, component):
        """Matrix H0 of one component, its lowest eigenpair, and all eigenpairs."""
        local, terms, li, lj, z = component
        H0 = _np.zeros((len(local), len(local)), complex)
        _np.add.at(H0, (li, lj), z * w[terms])
        _np.add.at(H0, (lj, li), z.conj() * w[terms])
        evals, evecs = _np.linalg.eigh(H0)
        return evals, evecs

    @staticmethod
    def _gauge_derivatives(evals, evecs, component, order):
        """t = -lambda_min(H0) and its first (and second) derivatives in the component's terms."""
        _, terms, li, lj, z = component
        t = -evals[0]
        if not order:
            return t, None, None
        if t <= 0:  # H0 = 0: g'(0) = 0, so the derivative of t never contributes
            return 0., _np.zeros(len(terms)), _np.zeros((len(terms), len(terms))) if order == 2 else None
        v0 = evecs[:, 0]
        # d(lambda)/dw = v0^dag E v0 with E = z e_i e_j^T + conj(z) e_j e_i^T.
        dt = -2 * (z * v0[li].conj() * v0[lj]).real
        if order == 1:
            return t, dt, None
        # V^dag E v0 for each term, then the standard second-order eigenvalue perturbation.
        P = evecs[li, :].conj().T * (z * v0[lj]) + evecs[lj, :].conj().T * (z.conj() * v0[li])
        gaps = evals[0] - evals[1:]
        keep = _np.abs(gaps) > 1e-12 * max(1., _np.abs(evals).max())  # within a repeated eigenvalue: no term
        Q = P[1:][keep] / _np.sqrt(-gaps[keep])[:, None]
        d2lambda = -2 * (Q.conj().T @ Q).real
        return t, dt, -d2lambda

    def _evaluate(self, blk, v, order=0):
        n = len(blk._bel_labels)
        if _np.shape(v) != (blk.num_params,) or not _np.all(_np.isfinite(v)):
            raise ValueError("Expected %d finite contraction-chart parameters." % blk.num_params)
        c, w = v[:n], v[n:]
        k, i, j, _ = blk._contraction_terms
        m, nd = len(w), len(blk._eeg_labels)
        g = _np.ones(m)
        data = _np.zeros(nd)
        data[blk._contraction_diagonal] = c * c
        J = _np.zeros((nd, len(v))) if order else None
        Hs = _np.zeros((nd, len(v), len(v))) if order == 2 else None
        if order:
            J[blk._contraction_diagonal, _np.arange(n)] = 2 * c
        if order == 2:
            Hs[blk._contraction_diagonal, _np.arange(n), _np.arange(n)] = 2.
        for component in blk._contraction_components:
            terms = component[1]
            evals, evecs = self._component_spectrum(blk, w, component)
            t, dt, d2t = self._gauge_derivatives(evals, evecs, component, order)
            q = 1 + t * t
            g[terms] = 2 / q
            if not order:
                continue
            g1 = -4 * t / q**2
            cc = c[i[terms]] * c[j[terms]]
            kk, pw = k[terms], n + terms
            # d(g w_a)/dw_b = g delta_ab + g' w_a dt_b
            dgw = g[terms][0] * _np.eye(len(terms)) + g1 * _np.outer(w[terms], dt)
            J[_np.ix_(kk, pw)] = cc[:, None] * dgw
            if order == 2:
                g2 = (12 * t * t - 4) / q**3
                gw = g[terms][0] * w[terms]
                # d2(g w_a)/dw_b dw_c = g'(dt_c delta_ab + dt_b delta_ac) + w_a (g'' dt_b dt_c + g' d2t_bc)
                second = w[terms][:, None, None] * (g2 * _np.outer(dt, dt) + g1 * d2t)[None]
                eye = _np.eye(len(terms))
                second += g1 * (eye[:, :, None] * dt[None, None, :] + eye[:, None, :] * dt[None, :, None])
                Hs[_np.ix_(kk, pw, pw)] = cc[:, None, None] * second
                # mixed c/w and c/c terms
                for row, a, b, ia, jb in zip(kk, range(len(terms)), terms, i[terms], j[terms]):
                    Hs[row, ia, pw] += c[jb] * dgw[a]
                    Hs[row, jb, pw] += c[ia] * dgw[a]
                    Hs[row, pw, ia] += c[jb] * dgw[a]
                    Hs[row, pw, jb] += c[ia] * dgw[a]
                    Hs[row, ia, jb] += gw[a]
                    Hs[row, jb, ia] += gw[a]
        data[k] = c[i] * c[j] * g * w
        if order:
            # derivatives with respect to the pivots c
            _np.add.at(J, (k, i), c[j] * g * w)
            _np.add.at(J, (k, j), c[i] * g * w)
        return data, J, Hs

    def params_to_block_data(self, blk, v):
        blk.block_data[:] = self._evaluate(blk, v)[0]

    def block_data_jacobian(self, blk, v):
        return self._evaluate(blk, v, order=1)[1]

    def superop_hessian(self, blk, superops, superops_are_flat, v=None):
        hessian = self._evaluate(blk, blk.to_vector() if v is None else v, order=2)[2]
        return _np.einsum('cpq,cij->ijpq', hessian, superops)

    def block_data_to_params(self, blk, ttol):
        n = len(blk._bel_labels)
        k, i, j, _ = blk._contraction_terms
        if not _np.all(_np.isfinite(blk.block_data)):
            raise ValueError("Contraction chart requires finite coefficients.")
        rates = blk.block_data[blk._contraction_diagonal]
        if _np.any(rates < ttol * max(1., _np.abs(rates).max())):
            raise ValueError("Contraction chart requires nonnegative S coefficients.")
        c = _np.sqrt(_np.clip(rates, 0, None))
        scale = c[i] * c[j]
        values = blk.block_data[k]
        orphan = (scale == 0) & (values != 0)
        if _np.any(orphan) and _np.isfinite(ttol):
            raise ValueError("A C/A coefficient is nonzero where one of its S coefficients is zero; "
                             "these coefficients are not positive semidefinite.")
        h = _np.divide(values, scale, out=_np.zeros(len(k)), where=scale > 0)  # entries of H
        tolerance = max(-ttol, 1e-9)
        w = _np.zeros(len(k))
        for component in blk._contraction_components:
            terms = component[1]
            evals, _ = self._component_spectrum(blk, h, component)
            gamma = max(-evals[0], 0.)
            if gamma > 1:
                if gamma > 1 + tolerance:
                    raise ValueError("Sparse S/C/A coefficients are not positive semidefinite "
                                     "(lambda_min of the normalized matrix is %g)." % (1 - gamma))
                h[terms] /= gamma
                gamma = 1.
            t = gamma / (1 + _np.sqrt(1 - gamma * gamma))  # root in [0, 1] of 2t / (1 + t^2) = gamma
            w[terms] = h[terms] * (1 + t * t) / 2
        return _np.r_[c, w]

    def param_labels(self, blk):
        k, i, j, active = blk._contraction_terms
        return (["sqrt[S(%s) coefficient]" % b for b in blk._bel_labels]
                + ["%s(%s,%s) contraction coordinate" % ('A' if a else 'C', blk._bel_labels[p], blk._bel_labels[q])
                   for p, q, a in zip(i, j, active)])

    def torch_stateless_data(self, blk, real_dtype, device):
        torch = _import_torch()
        index = lambda a: torch.tensor(_np.asarray(a), dtype=torch.long, device=device)
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        k, i, j, _ = blk._contraction_terms
        components = tuple((len(local), index(terms), index(li), index(lj),
                            torch.tensor(z, dtype=complex_dtype, device=device))
                           for local, terms, li, lj, z in blk._contraction_components)
        return (len(blk._bel_labels), len(blk._eeg_labels), index(blk._contraction_diagonal),
                index(k), index(i), index(j), components)

    @staticmethod
    def block_data_torch(sd, t_param):
        torch = _import_torch()
        n, nd, diagonal, k, i, j, components = sd
        c, w = t_param[:n], t_param[n:]
        g = t_param.new_ones(len(w))
        for size, terms, li, lj, z in components:
            entries = z * w[terms].to(z.dtype)
            H0 = z.new_zeros((size, size)).index_put((li, lj), entries, accumulate=True)
            H0 = H0.index_put((lj, li), entries.conj(), accumulate=True)
            t = -torch.linalg.eigvalsh(H0)[0]
            g = g.index_put((terms,), (2 / (1 + t * t)).to(g.dtype).expand(len(terms)))
        data = t_param.new_zeros(nd).index_put((diagonal,), c * c)
        return data.index_put((k,), c[i] * c[j] * g * w)

    def _coefficient_polynomial(self, blk, index, pio, mpv):
        raise NotImplementedError("The contraction chart is nonpolynomial; use matrix simulation or Torch, "
                                  "rather than polynomial term simulation.")


class LindbladCoefficientBlock(_NicelySerializable):
    """ 
    Class for storing and managing the parameters associated with particular subblocks of error-generator
    parameters. Responsible for management of different internal representations utilized when employing
    various error generator constraints.
    """

    _superops_cache = {}  # a custom cache for create_lindblad_term_superoperators method calls

    # Registry mapping block_type string -> concrete subclass; populated at end of this module.
    _BLOCK_TYPES = {}

    def __new__(cls, block_type=None, *args, **kwargs):
        # When the base class is instantiated directly -- the public/back-compat entry point, e.g.
        # ``LindbladCoefficientBlock('ham', ...)`` -- construct the concrete subclass registered for
        # the requested ``block_type``.  Subclasses (and pickle/deepcopy, which invoke
        # ``cls.__new__(cls)`` with no ``block_type``) allocate normally without dispatching.
        if cls is LindbladCoefficientBlock:
            try:
                cls = LindbladCoefficientBlock._BLOCK_TYPES[block_type]
            except KeyError:
                raise InvalidBlockTypeError()
        return super().__new__(cls)

    def __init__(self, block_type, basis, basis_element_labels=None, initial_block_data=None, param_mode='static',
                 truncate=False, *, error_generator_labels=None, cholesky_ordering=None,
                 completion_normalization='row'):
        """
        Parameters
        ----------
        block_type : str
            String specifying the type of error generator parameters contained within this block. Allowed
            values are 'ham' (for Hamiltonian error generators), 'other_diagonal' (for Pauli stochastic error generators),
            'other' (for S-type stochastic, C-type correlation, and A-type active error generators), and
            'other_sparse' (selected S coefficients and C/A pair terms with sparse CPTP charts).
        
        basis : `Basis`
            `Basis` object to be used by this coefficient block. Note this must be an actual `Basis` object, and not
            a string (as the coefficient block doesn't have the requisite dimensionality information needed for casting).
        
        basis_element_labels : list or tuple of str
            Iterable of strings corresponding to the basis element subscripts used by the error generators managed by
            this coefficient block.
        
        initial_block_data : _np.ndarray, optional (default None)
            Numpy array with initial parameter values to use in setting initial state of this coefficient block.
        
        param_mode : str, optional (default 'static')
            String specifying the type of internal parameterization used by this coefficient block. Allowed options are:
            
            - For all block types: 'static'
            - For 'ham': 'elements'
            - For 'other_diagonal': 'elements', 'cholesky', 'depol', 'reldepol'
            - For 'other': 'elements', 'cholesky'
            - For 'other_sparse': 'elements', 'cholesky', 'contraction', 'cholesky completion', 'auto'

            The fill-free 'cholesky' chart includes zero noise and singular PSD matrices.
            'contraction' writes K = C (I + H) C with C the square roots of the S
            coefficients and H a scaled contraction on the selected C/A components. It
            covers every PSD matrix on any support, including zero noise and the
            boundary, without completion, and does not support polynomial term simulation.
            'cholesky completion' covers the positive-definite interior on arbitrary
            support and independent C/A masks; its diagonal coordinates are logarithmic
            and free off-diagonal coordinates are divided by the row pivot. It starts
            at an identity Kossakowski matrix when no initial data is supplied. Singular
            inputs are rejected, including when `truncate` is requested. This chart does
            not support polynomial term simulation. 'auto' chooses the fill-free chart
            when possible and the contraction chart otherwise.

            Note that the most commonly encounted settings in practice are 'elements' and 'cholesky',
            which when used in the right combination are utilized in the construction of GLND and CPTPLND
            parameterized models. For both GLND and CPTPLND the 'ham' block used the 'elements' `param_mode`.
            GLND the 'other' block uses 'elements', and for CPTPLND it uses 'cholesky'. 
            
            'depol' and 'reldepol' are special modes used only for Pauli stochastic only coefficient blocks
            (i.e. 'other_diagonal'), and correspond to special reduced parameterizations applicable to depolarizing
            channels. (TODO: Add better explanation of the difference between depol and reldepol).
        
        truncate : bool, optional (default False)
            Flag specifying whether to truncate the parameters given by `initial_block_data` in order to meet
            constraints (e.g. to preserve CPTP) when necessary. If False, then an error is thrown when the
            given intial data cannot be parameterized as specified.

        error_generator_labels : list or tuple of `LocalElementaryErrorgenLabel`, optional (default None)
            Keyword-only.  Only supported by block types whose `block_data` is keyed directly by elementary
            error generators ('other_unconstrained' and 'other_sparse'); mutually exclusive with `basis_element_labels`.
            When given, the block manages exactly the listed elementary error generators, enabling reduced /
            flexible parameterizations.  Must be `None` for the other (basis-element-keyed) block types.

        cholesky_ordering : array_like, optional
            Only for 'other_sparse': an elimination permutation in S-direction order.
            Completion accepts any permutation; the fill-free chart requires no fill.
            The resolved ordering is saved with the block to preserve parameter meaning
            across machines with different ordering backends.

        completion_normalization : {'row', 'column'}, optional
            For completion, normalize free factor entries by their row pivot (default),
            giving a unit-diagonal completion without pivot divisions. Column normalization
            can be better conditioned near other PSD boundaries, and also retains compatibility
            with earlier development serializations.
        """

        super().__init__()
        self._block_type = block_type  # 'ham' or 'other' or 'other_diagonal'
        self._param_mode = param_mode  # 'static', 'elements', 'cholesky', 'depol', 'reldepol'
        _pcls = type(self)._PARAM_CLASSES.get(param_mode)
        if _pcls is None:
            raise InvalidParamModeError(param_mode, block_type)
        self._parameterization = _pcls()  # composition: owns the param_mode-specific maps
        self._basis = basis  # must be a full Basis object, not just a string, as we otherwise don't know dimension
        if cholesky_ordering is not None and block_type != 'other_sparse':
            raise ValueError("`cholesky_ordering` is only supported by 'other_sparse' blocks.")
        self._cholesky_ordering = cholesky_ordering
        if completion_normalization not in ('row', 'column'):
            raise ValueError("`completion_normalization` must be 'row' or 'column'.")
        self._completion_normalization = completion_normalization
        self._init_labels(basis, basis_element_labels, error_generator_labels)
        self._cache_mx = _np.zeros((len(self._bel_labels), len(self._bel_labels)), 'complex') \
            if self._uses_cache_mx else None

        #this would get set to True in the very next method call anyway
        self._coefficients_need_update = True
        self._cached_elementary_errorgens = None
        self._cached_elementary_errorgen_indices = None

        self._set_block_data(initial_block_data, truncate)

    def _init_labels(self, basis, basis_element_labels, error_generator_labels):
        # Default: this block's coefficients are keyed by basis element labels and it has no explicit
        # elementary-error-generator label list.  Subclasses whose block_data is keyed directly by
        # elementary error generators (e.g. _OtherUnconstrainedCoeffBlock) override this.
        if error_generator_labels is not None:
            raise ValueError("`error_generator_labels` is not supported by block type '%s'!" % self._block_type)
        self._bel_labels = tuple(basis_element_labels) if (basis_element_labels is not None) \
            else tuple(basis.labels[1:])  # Note: don't include identity
        self._eeg_labels = None

    def _coeff_count(self):
        """Number of entries along each axis of ``block_data`` (block_data has ``_coeff_ndim`` such axes)."""
        return len(self._bel_labels)

    def _set_block_data(self, block_data, truncate):
        #Sets self.block_data directly, which may later be overwritten by to/from vector calls so this
        # is somewhat dangerous to call, thus a private member.  It should really only be used during
        # initialization and to set/update a param_mode=='static' block.
        # Note: block_data == None can be used to initialize all-zero parameters
        n = self._coeff_count()
        block_shape = (n,) if self._coeff_ndim == 1 else (n, n)
        block_dtype = self._block_dtype

        if block_data is not None:
            assert(block_data.shape == block_shape), \
                "Invalid `initial_block_data` shape: expected %s, received %s" % (
                    str(block_shape), str(block_data.shape))
            assert(block_data.dtype == _np.dtype(block_dtype)), \
                "Invalid `initial_block_data` dtype: expected %s, received %s" % (
                    str(block_dtype), str(block_data.dtype))
            self.block_data = _np.ascontiguousarray(block_data)
        else:
            self.block_data = _np.zeros(block_shape, block_dtype)

        self._truncate_block_data(truncate)

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

    @property
    def basis_element_labels(self):
        return self._bel_labels

    @property
    def block_type(self):
        """The block-type string: 'ham', 'other_diagonal', or 'other'."""
        return self._block_type

    @property
    def _coeff_shape(self):
        """Shape of this block's ``block_data`` array."""
        n = self._coeff_count()
        return (n,) if self._coeff_ndim == 1 else (n, n)

    # --- block-type-specific hooks (implemented by the subclasses in _BLOCK_TYPES) ---

    @property
    def num_params(self) -> int:
        return self._parameterization.num_params(self)

    def create_lindblad_term_superoperators(self, mx_basis='pp', sparse: Union[Literal['auto'], bool]="auto", include_1norms=False, flat=False):
        assert(self._basis is not None), "Cannot create lindblad superoperators without a basis!"
        basis = self._basis
        sparse = basis.sparse if (sparse == "auto") else sparse
        mxs = [basis[lbl] for lbl in self._bel_labels]
        if len(mxs) == 0:
            return ([], []) if include_1norms else []  # short circuit - no superops to return

        d = mxs[0].shape[0]
        d2 = d**2  # if mxs[0] is sparse, then d2 != mxs[0].size.
        mx_basis = _Basis.cast(mx_basis, d2, sparse=sparse)

        cache_key = (self._block_type, tuple(self._bel_labels), mx_basis, basis)
        if cache_key not in self._superops_cache:
            self._update_superops_cache(mxs, mx_basis, cache_key)
        cached_superops, cached_superops_1norms = self._superops_cache[cache_key]

        if flat or not self._superops_matrix_structured:
            if include_1norms:
                return _copy.deepcopy(cached_superops), cached_superops_1norms.copy()
            else:
                return _copy.deepcopy(cached_superops)

        out = _unflatten_cached_other_lindblad_term_superoperators(cached_superops, cached_superops_1norms, include_1norms)
        return out

    def _update_superops_cache(self, mxs, mx_basis, cache_key):
        d2 = mx_basis.dim
        sparse = mx_basis.sparse
        superops = self._compute_term_generators(mxs, d2, sparse)  # in std basis

        # Convert matrices to desired `mx_basis` basis
        if mx_basis != "std":
            superops = _custom_superops_stdbasis_conversion(mx_basis, sparse, superops)
        
        superop_1norms = _np.array([_mt.safe_onenorm(mx) for mx in superops], 'd')

        self._superops_cache[cache_key] = (superops, superop_1norms)
        return 

    def create_lindblad_term_objects(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        return self._create_lindblad_term_objects_impl(evotype, state_space, max_polynomial_vars,
                                                       parameter_index_offset)

    @property
    def elementary_errorgen_indices(self):
        """
        TODO docstring - rewrite this docstring - especially return value!
        Constructs a  dictionary mapping Lindblad term labels to projection coefficients.

        This method is used for finding the index of a particular error generator coefficient
        in the 1D array formed by concatenating the Hamiltonian and flattened stochastic
        projection arrays.

        Parameters
        ----------
        ham_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct `ham_projs`.  Allowed values are Matrix-unit
            (std), Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt), list of
            numpy arrays, or a custom basis object.

        other_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct `other_projs`.  Allowed values are
            Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt),
            list of numpy arrays, or a custom basis object.

        other_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad error projections `other_projs` includes.
            Allowed values are: `"diagonal"` (only the diagonal Stochastic),
            `"diag_affine"` (diagonal + affine generators), and `"all"`
            (all generators).

        Returns
        -------
        elem_errgen_indices : dict
            Keys are `LocalElementaryErrorgenLabel` objects. H-type Hamiltonian and
            S-type stochastic terms have one operator-basis label; C-type correlation
            and A-type active pair terms have two distinct labels. Values are lists of
            (multiplier, flattened block index) pairs that express each elementary
            coefficient as a linear combination of the block's coefficients.
        """
        # Note: returned dictionary's value specify a linear combination of
        # this coefficient block's coefficients that product the given (by the key)
        # elementary error generator.  Values are lists of (c_i, index_i) pairs,
        # such that the given elementary generator == sum_i c_i * coefficients_in_flattened_block[index_i]
        if not self._coefficients_need_update and self._cached_elementary_errorgen_indices is not None:
            return self._cached_elementary_errorgen_indices

        idx = self._elementary_errorgen_indices_impl()

        self._cached_elementary_errorgen_indices = idx
        return idx

    @property
    def _block_data_indices(self):
        """
        Effectively the inverse of elementary_errorgen_indices.

        The keys of the returned dict are (flattened) block_data indices and the
        values specify a linear combination of elementary errorgens via their labels.
        """
        return self._block_data_indices_impl()

    @property
    def elementary_errorgens(self):
        """
        Converts a set of coefficients for this block into a linear combination of elementary error generators.

        This linear combination is given by a dictionary with keys equal to elementary
        error generator labels and values equal to their coefficients in the linear combination.

        Parameters
        ----------
        block_data : numpy.ndarray
            A 1- or 2-dimensional array with each dimension of size `len(self.basis_element_labels)`,
            specifying the coefficients of this block.  Array is 1-dimensional when this block is
            of type `'ham'` or `'other_diagonal'` and is 2-dimensional for type `'other'`.

        Returns
        -------
        elementary_errorgens : dict
            Specifies `block_data` as a linear combination of elementary error generators.
            Keys are :class:`LocalElementaryErrorgenLabel` objects and values are floats.
        """
        if not self._coefficients_need_update and self._cached_elementary_errorgens is not None:
            return self._cached_elementary_errorgens

        elementary_errorgens = dict()
        eeg_indices = self.elementary_errorgen_indices
        flat_data = self.block_data.ravel()

        for eeg_lbl, linear_combo in eeg_indices.items():
            val = _np.sum([coeff * flat_data[index] for coeff, index in linear_combo])
            elementary_errorgens[eeg_lbl] = _np.real_if_close(val).item()  # item() -> scalar
            #set_basis_el(lbl, basis[lbl])  # REMOVE
        #cache the error generator dictionary for future use
        self._cached_elementary_errorgens = elementary_errorgens
        self._coefficients_need_update = False

        return elementary_errorgens

    def set_elementary_errorgens(self, elementary_errorgens, on_missing='ignore', truncate=False):
        # Note: could return a "stripped" version of elementary_errorgens
        # expects a dict with keys == local elem errgen labels
        flat_data = self.block_data.ravel().copy()
        unused_elementary_errorgens = elementary_errorgens.copy()  # to return
        for i, linear_combo in self._block_data_indices.items():
            val = 0
            for coeff, eeg_lbl in linear_combo:
                if eeg_lbl in elementary_errorgens:
                    val += coeff * elementary_errorgens[eeg_lbl]
                    if eeg_lbl in unused_elementary_errorgens:
                        del unused_elementary_errorgens[eeg_lbl]
                elif on_missing == 'warn':
                    _warnings.warn("Missing entry for %s in dictionary of elementary errorgens.  Assuming 0."
                                   % str(eeg_lbl))
                elif on_missing == 'raise':
                    raise ValueError("Missing entry for %s in dictionary of elementary errorgens." % str(eeg_lbl))
            flat_data[i] = val

        self.block_data[:] = flat_data.reshape(self.block_data.shape)
        self._truncate_block_data(truncate)

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

        return unused_elementary_errorgens

    def set_from_errorgen_projections(self, errorgen, errorgen_basis='pp', return_projected_errorgen=False,
                                      truncate=False):
        elementary_errorgen_lbls = list(self.elementary_errorgens.keys())
        out = _ot.extract_elementary_errorgen_coefficients(errorgen, elementary_errorgen_lbls,
                                                           self._basis, errorgen_basis,
                                                           return_projected_errorgen)
        elementary_errorgens = out[0] if return_projected_errorgen else out
        unused = self.set_elementary_errorgens(elementary_errorgens, on_missing='raise', truncate=truncate)
        assert(len(unused) == 0)

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

        return out[1] if return_projected_errorgen else None

    @property
    def param_labels(self) -> list:
        """
        Generate human-readable labels for the parameters of this block.
        """
        return self._parameterization.param_labels(self)

    def _block_data_to_params(self, truncate: Union[bool, float]=False):
        """
        Compute parameter values implied by self.block_data.
        (The public method now simply dispatches on block type.)
        """
        if truncate is False:
            ttol = -1e-14
        elif truncate is True:
            ttol = -_np.inf
        else:
            ttol = -truncate

        if self._param_mode == 'static':
            return _np.empty(0, 'd')

        params = self._parameterization.block_data_to_params(self, ttol)

        assert(not _np.iscomplexobj(params))
        assert(len(params) == self.num_params)
        return params

    def _truncate_block_data(self, truncate=False):
        if truncate is False: return
        v = self._block_data_to_params(truncate)
        self.from_vector(v)

    def to_vector(self):
        """
        Compute parameter values for this coefficient block.

        Returns
        -------
        param_vals : numpy.ndarray
            A 1D array of real parameter values. Length is `len(self.basis_element_labels)`
            in the case of `'ham'` or `'other_diagonal'` blocks, and `len(self.basis_element_labels)**2`
            in the case of `'other'` blocks.
        """
        return self._block_data_to_params(truncate=False)

    def from_vector(self, v: _np.ndarray):
        """
        Construct Lindblad coefficients (for this block) from a set of parameter values.
        This is now just a dispatcher into the three block-type specific implementations.
        """
        # static blocks carry no parameters
        if self._param_mode == 'static':
            assert len(v) == 0, "'static' parameterized blocks should have zero parameters!"
            return

        self._parameterization.params_to_block_data(self, v)

        # mark cache stale
        self._coefficients_need_update = True
        return

    def deriv_wrt_params(self, v=None):
        """
        Jacobian of block_data w.r.t. the real parameters v.
        Dispatches to _deriv_wrt_params_<block_type>.
        """
        v = self.to_vector() if (v is None) else v
        nP = len(v)
        assert(nP == self.num_params), f"Expected {self.num_params} parameters, got {nP}!"

        # static blocks have zero‐dimensional parameter space
        if self._param_mode == 'static':
            return _np.zeros(self._coeff_shape + (0,), 'd')

        return self._parameterization.block_data_jacobian(self, v)

    def elementary_errorgen_deriv_wrt_params(self, v=None):
        eeg_indices = self.elementary_errorgen_indices
        blkdata_deriv = self.deriv_wrt_params(v)
        if blkdata_deriv.ndim == 3:  # (coeff_dim_1, coeff_dim_2, param_dim) => (coeff_dim, param_dim)
            cd1, cd2, pd = blkdata_deriv.shape  # type: ignore
            blkdata_deriv = blkdata_deriv.reshape((cd1 * cd2, pd), order='C')  # blkdata_deriv rows <=> flat_data indices

        eeg_deriv = _np.zeros((len(eeg_indices), self.num_params), 'd')  # may need to be complex?

        # Note: ordering in eeg_indices matches that of self.elementary_errorgens (as it must for this to be correct)
        for i, (eeg_lbl, linear_combo) in enumerate(eeg_indices.items()):
            deriv = _np.sum([coeff * blkdata_deriv[index, :] for coeff, index in linear_combo], axis=0)
            eeg_deriv[i, :] = _np.real_if_close(deriv)
        return eeg_deriv

    def superop_deriv_wrt_params(self, superops : _np.ndarray, v=None, superops_are_flat=False) -> _np.ndarray:
        """
        Derivative of the Lindblad-term superoperators w.r.t. the real parameters.

        Parameters
        ----------
        superops : ndarray
            The output of create_lindblad_term_superoperators(..., sparse=False).

        v : array, optional
            The parameter vector. Only needed for 'other_diagonal' blocks.
            
        superops_are_flat : bool, optional
            True if `superops` is a flat list/array even for the 'other' block.
        """
        # --- static case: always zero ---
        if self._param_mode == 'static':
            if superops_are_flat or not self._superops_matrix_structured:
                return _np.zeros((superops.shape[1], superops.shape[2], 0), 'd')
            else:
                return _np.zeros((superops.shape[2], superops.shape[3], 0), 'd')

        # The error generator is linear in block_data (superop = sum_i block_data_flat[i] * G_i), so by
        # the chain rule  d(superop)/dv = einsum('Dp,Djk->jkp', d(block_data)/dv, G).  Only the
        # parameterization's Jacobian is param_mode-specific
        v = self.to_vector() if (v is None) else v
        nP = len(v)
        if self._superops_matrix_structured and not superops_are_flat:
            n = len(self._bel_labels)
            Gflat = superops.reshape((n * n, superops.shape[2], superops.shape[3]))
        else:
            Gflat = superops
        Jflat = self.deriv_wrt_params(v).reshape(-1, nP)
        dOdp = _np.einsum('Dp,Djk->jkp', Jflat, Gflat)
        if self._superops_matrix_structured:
            n = len(self._bel_labels)
            dOdp = dOdp.reshape(dOdp.shape[0], dOdp.shape[1], n, n)  # legacy (d2,d2,n,n) param shape

        # --- common checks & real‐cast ---
        tr = dOdp.ndim
        assert (tr - 2) in (1, 2), "Currently, dodp can only have 1 or 2 derivative dimensions"
        assert _np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL
        return _np.real(dOdp)

    def superop_hessian_wrt_params(self, superops, v=None, superops_are_flat=False):
        # Completed Cholesky has parameter-dependent second derivatives.

        # static blocks always have zero Hessian
        if self._param_mode == 'static':
            if superops_are_flat or not self._superops_matrix_structured:
                return _np.zeros((superops.shape[1], superops.shape[2], 0, 0), 'd')
            else:
                return _np.zeros((superops.shape[2], superops.shape[3], 0, 0), 'd')

        # dispatch to block‐type helper
        d2 = self._parameterization.superop_hessian(self, superops, superops_are_flat, v=v)

        # sanity checks & real‐cast
        tr = d2.ndim
        assert (tr - 2) in (2, 4), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"
        assert _np.linalg.norm(_np.imag(d2)) < IMAG_TOL
        return _np.real(d2)

    def stateless_data(self, real_dtype, device):
        """
        Returns `(parameterization_class, torch_ctx)`, where `torch_ctx` carries the constants that
        `torch_base` needs.  This mirrors the Torchable API, but lives on the (non-ModelMember)
        coefficient block, so that a Torchable `LindbladErrorgen` can compose its blocks' contributions.
        """
        inner_sd = self._parameterization.torch_stateless_data(self, real_dtype, device)
        return (type(self._parameterization), inner_sd)

    @staticmethod
    def torch_base(sd, t_param):
        """Differentiable reconstruction of `block_data` from the parameter tensor `t_param`.

        With `sd = blk.stateless_data(...)` and `t_param = torch.from_numpy(blk.to_vector())`,
        `type(blk).torch_base(sd, t_param)` reproduces `blk.block_data` as a torch Tensor whose
        autodiff Jacobian w.r.t. `t_param` equals `blk.deriv_wrt_params(...)`. 
        """
        param_cls, inner_sd = sd
        return param_cls.block_data_torch(inner_sd, t_param)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        # Always serialize under the *base* class name; ``block_type`` (below) selects the concrete
        # subclass on load (via __new__ in _from_nice_serialization).  This keeps the on-disk format
        # stable and backward/forward compatible across the base+subclasses refactor.
        state['module'] = LindbladCoefficientBlock.__module__
        state['class'] = LindbladCoefficientBlock.__name__
        state.update({'block_type': self._block_type,
                      'parameterization_mode': self._param_mode,
                      'basis_element_labels': list(self._bel_labels),
                      'basis': self._basis.to_nice_serialization(),
                      'block_data': self._encodemx(self.block_data)})
        if self._eeg_labels is not None:
            # Blocks keyed directly by elementary error generators (e.g. 'other_unconstrained') can't
            # be reconstructed from basis_element_labels alone (reduced/subset blocks), so store the
            # explicit eeg labels.  ``class``/``module`` stay pinned to the base name (set above), so
            # this dict is read back below regardless of the concrete subclass.
            state['error_generator_labels'] = [[lbl.errorgen_type, list(lbl.basis_element_labels)]
                                               for lbl in self._eeg_labels]
        if self._block_type == 'other_sparse' and hasattr(self, 'cholesky_structure'):
            state['cholesky_ordering'] = self.cholesky_structure.perm.tolist()
        if self._param_mode == 'cholesky completion':
            state['completion_normalization'] = self._completion_normalization
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        block_data = cls._decodemx(state['block_data'])
        basis = _Basis.from_nice_serialization(state['basis'])
        if state.get('error_generator_labels', None) is not None:
            eeg_labels = [_LEEL(etype, tuple(bels)) for etype, bels in state['error_generator_labels']]
            return cls(state['block_type'], basis, initial_block_data=block_data,
                       param_mode=state['parameterization_mode'], error_generator_labels=eeg_labels,
                       cholesky_ordering=state.get('cholesky_ordering'),
                       completion_normalization=state.get('completion_normalization', 'column'))
        return cls(state['block_type'], basis, state['basis_element_labels'], block_data,
                   state['parameterization_mode'])

    def is_similar(self, other_coeff_block):
        """ TODO: docstring """
        if not isinstance(other_coeff_block, LindbladCoefficientBlock): return False
        return ((self._block_type == other_coeff_block._block_type)
                and (self._param_mode == other_coeff_block._param_mode)
                and (self._bel_labels == other_coeff_block._bel_labels)
                and (self._basis == other_coeff_block._basis))

    def convert(self, param_mode):
        """
        TODO: docstring  - return a *new* LindbladCoefficientBlock with the same block type and data,
        but with the given parameterization mode.
        """
        return LindbladCoefficientBlock(self._block_type, self._basis, self._bel_labels, self.block_data, param_mode)

    def __str__(self):
        s = '%s-type lindblad coefficient block with param-mode %s and basis labels %s.' % (
            self._block_type, self._param_mode, self._bel_labels)
        if len(self._bel_labels) < 10:
            s += " Coefficients are:\n" + str(_np.round(self.block_data, 4))
        return s


class _HamCoeffBlock(LindbladCoefficientBlock):
    """Hamiltonian (``block_type='ham'``) Lindblad coefficient block.

    ``block_data`` is a length-n real vector of Hamiltonian-generator coefficients.
    Valid param modes: 'static', 'elements'.
    """
    _coeff_ndim = 1
    _block_dtype = 'd'
    _uses_cache_mx = False
    _superops_matrix_structured = False
    _PARAM_CLASSES = {'static': _StaticParam, 'elements': _HamElements}

    def _compute_term_generators(self, mxs, d2, sparse):
        nMxs = len(mxs)
        superops = [None] * nMxs if sparse else _np.empty((nMxs, d2, d2), 'complex')
        for i, B in enumerate(mxs):
            superops[i] = _lt.create_lindbladian_term_errorgen('H', B, sparse=sparse)  # in std basis
        return superops

    def _create_lindblad_term_objects_impl(self, evotype, state_space, mpv, pio):
        Lterms = []
        for k, bel_label in enumerate(self._bel_labels):  # k == index of local parameter that is coefficient
            # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
            scale, U = _mt.to_unitary(self._basis[bel_label])
            coeff_poly = self._parameterization._coefficient_polynomial(self, k, pio, mpv)

            U_dag = U.conjugate().T

            # Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
            poly_left = (-1j * scale) * coeff_poly
            term_left = _term.RankOnePolynomialOpTerm.create_from(
                poly_left, U, None, evotype, state_space)

            poly_right = (1j * scale) * coeff_poly
            term_right = _term.RankOnePolynomialOpTerm.create_from(
                poly_right, None, U_dag, evotype, state_space)

            Lterms.extend([term_left, term_right])
        return Lterms

    def _elementary_errorgen_indices_impl(self):
        elem_errgen_indices = _collections.OrderedDict()
        for i, lbl in enumerate(self._bel_labels):
            elem_errgen_indices[_LEEL('H', (lbl,))] = [(1.0, i)]
        return elem_errgen_indices

    def _block_data_indices_impl(self):
        block_data_indices = dict()
        for i, lbl in enumerate(self._bel_labels):
            block_data_indices[i] = [(1.0, _LEEL('H', (lbl,)))]
        return block_data_indices

    def _coefficient_labels_impl(self):
        return ["%s Hamiltonian error coefficient" % lbl for lbl in self._bel_labels]


class _OtherDiagonalCoeffBlock(LindbladCoefficientBlock):
    """Pauli-stochastic diagonal (``block_type='other_diagonal'``) Lindblad coefficient block.

    ``block_data`` is a length-n real vector of diagonal (Pauli-stochastic) coefficients.
    Valid param modes: 'static', 'elements', 'cholesky', 'depol', 'reldepol'.
    """
    _coeff_ndim = 1
    _block_dtype = 'd'
    _uses_cache_mx = False
    _superops_matrix_structured = False
    _PARAM_CLASSES = {'static': _StaticParam, 'elements': _DiagElements, 'cholesky': _DiagCholesky,
                      'depol': _Depol, 'reldepol': _RelDepol}

    def _compute_term_generators(self, mxs, d2, sparse):
        nMxs = len(mxs)
        superops = [None] * nMxs if sparse else _np.empty((nMxs, d2, d2), 'complex')
        for i, Lm in enumerate(mxs):
            superops[i] = _lt.create_lindbladian_term_errorgen('O', Lm, Lm, sparse)  # in std basis
        return superops

    def _create_lindblad_term_objects_impl(self, evotype, state_space, mpv, pio):
        Lterms = []
        for k, bel_label in enumerate(self._bel_labels):  # k == index of local parameter that is coefficient
            # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
            scale, U = _mt.to_unitary(self._basis[bel_label])
            scale = scale**2  # because there are two "U"s in each overall term below

            base_poly = self._parameterization._coefficient_polynomial(self, k, pio, mpv) * scale
            Lm = Ln = U
            Lm_dag = Lm.conjugate().T  # assumes basis is dense (TODO: make sure works
            Ln_dag = Ln.conjugate().T  # for sparse case too - and np.dots below!)

            # Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
            # e.g. in 2nd term, _np.dot(Ln_dag, Lm) == adjoint(_np.dot(Lm_dag,Ln))
            poly_sandwich = 1.0 * base_poly
            term_sandwich = _term.RankOnePolynomialOpTerm.create_from(
                poly_sandwich, Ln, Lm, evotype, state_space)

            poly_anti = -0.5 * base_poly
            op_post = _np.dot(Ln_dag, Lm)
            term_post = _term.RankOnePolynomialOpTerm.create_from(
                poly_anti, None, op_post, evotype, state_space)

            op_pre = _np.dot(Lm_dag, Ln)
            term_pre = _term.RankOnePolynomialOpTerm.create_from(
                poly_anti, op_pre, None, evotype, state_space)

            Lterms.extend([term_sandwich, term_post, term_pre])
        return Lterms

    def _elementary_errorgen_indices_impl(self):
        elem_errgen_indices = _collections.OrderedDict()
        for i, lbl in enumerate(self._bel_labels):
            elem_errgen_indices[_LEEL('S', (lbl,))] = [(1.0, i)]
        return elem_errgen_indices

    def _block_data_indices_impl(self):
        block_data_indices = dict()
        for i, lbl in enumerate(self._bel_labels):
            block_data_indices[i] = [(1.0, _LEEL('S', (lbl,)))]
        return block_data_indices

    def _coefficient_labels_impl(self):
        return ["(%s,%s) other error coefficient" % (lbl, lbl) for lbl in self._bel_labels]


class _OtherCoeffBlock(LindbladCoefficientBlock):
    """Full non-Hamiltonian (``block_type='other'``) Lindblad coefficient block with
    S-type stochastic, C-type correlation, and A-type active error generators.

    ``block_data`` is an n x n complex Hermitian coefficient matrix.
    Valid param modes: 'static', 'elements', 'cholesky'.
    """
    _coeff_ndim = 2
    _block_dtype = 'complex'
    _uses_cache_mx = True
    _superops_matrix_structured = True
    _PARAM_CLASSES = {'static': _StaticParam, 'elements': _OtherElements, 'cholesky': _OtherCholesky}

    def _compute_term_generators(self, mxs, d2, sparse):
        nMxs = len(mxs)
        superops = [None] * nMxs**2 if sparse else _np.empty((nMxs**2, d2, d2), 'complex')
        for i, Lm in enumerate(mxs):
            for j, Ln in enumerate(mxs):
                superops[i * nMxs + j] = _lt.create_lindbladian_term_errorgen('O', Lm, Ln, sparse)
        return superops

    def _create_lindblad_term_objects_impl(self, evotype, state_space, mpv, pio):
        Lterms = []
        for i, bel_labeli in enumerate(self._bel_labels):
            for j, bel_labelj in enumerate(self._bel_labels):
                scalem, Um = _mt.to_unitary(self._basis[bel_labeli])  # ensure all Rank1Term operators are *unitary*
                scalen, Un = _mt.to_unitary(self._basis[bel_labelj])  # ensure all Rank1Term operators are *unitary*
                Lm, Ln = Um, Un
                Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                scale = scalem * scalen

                coeff_poly = self._parameterization._coefficient_polynomial(self, (i, j), pio, mpv)
                base_poly = coeff_poly * scale

                # Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
                poly_sandwich = 1.0 * base_poly
                term_sandwich = _term.RankOnePolynomialOpTerm.create_from(
                    poly_sandwich, Ln, Lm, evotype, state_space)

                poly_anti = -0.5 * base_poly
                op_post = _np.dot(Ln_dag, Lm)  # adjoint(dot(Lm_dag,Ln))
                term_post = _term.RankOnePolynomialOpTerm.create_from(
                    poly_anti, None, op_post, evotype, state_space)

                op_pre = _np.dot(Lm_dag, Ln)
                term_pre = _term.RankOnePolynomialOpTerm.create_from(
                    poly_anti, op_pre, None, evotype, state_space)

                Lterms.extend([term_sandwich, term_post, term_pre])
        return Lterms

    def _elementary_errorgen_indices_impl(self):
        elem_errgen_indices = _collections.OrderedDict()
        # Difficult case, as coefficients do not correspond to elementary errorgens, so
        # there's no single index for, e.g. ('C', lbl1, lbl2) - rather this elementary
        # errorgen is a linear combination of two coefficients.
        stride = len(self._bel_labels)
        for i, lbl1 in enumerate(self._bel_labels):
            ii = i * stride + i
            elem_errgen_indices[_LEEL('S', (lbl1,))] = [(1.0, ii)]

            for j, lbl2 in enumerate(self._bel_labels[i + 1:], start=i + 1):
                ij = i * stride + j
                ji = j * stride + i

                # NH_PQ = (C_PQ + i A_PQ)/2 ; NH_QP = (C_PQ - i A_PQ)/2
                elem_errgen_indices[_LEEL('C', (lbl1, lbl2))] = [(0.5, ij), (0.5, ji)]  # C_PQ contributions
                elem_errgen_indices[_LEEL('A', (lbl1, lbl2))] = [(0.5j, ij), (-0.5j, ji)]  # A_PQ contributions
        return elem_errgen_indices

    def _block_data_indices_impl(self):
        # Difficult case, as coefficients do not correspond to elementary errorgens.
        block_data_indices = dict()
        stride = len(self._bel_labels)
        for i, lbl1 in enumerate(self._bel_labels):
            ii = i * stride + i
            block_data_indices[ii] = [(1.0, _LEEL('S', (lbl1,)))]
            for j, lbl2 in enumerate(self._bel_labels[i + 1:], start=i + 1):
                ij = i * stride + j
                ji = j * stride + i
                # C_PQ = NH_PQ + NH_QP ;  A_PQ = i(NH_QP - NH_PQ)
                block_data_indices[ij] = [(1.0, _LEEL('C', (lbl1, lbl2))), (-1.0j, _LEEL('A', (lbl1, lbl2)))]
                block_data_indices[ji] = [(1.0, _LEEL('C', (lbl1, lbl2))), (+1.0j, _LEEL('A', (lbl1, lbl2)))]
        return block_data_indices


class _OtherUnconstrainedCoeffBlock(LindbladCoefficientBlock):
    """Flat, unconstrained non-Hamiltonian (``block_type='other_unconstrained'``) Lindblad
    coefficient block.

    Unlike ``_OtherCoeffBlock`` (which stores an n x n Hermitian coefficient matrix), this block
    stores ``block_data`` as a flat 1-D real vector in one-to-one correspondence with an explicit
    list of elementary error generators (each of type 'S', 'C', or 'A').  This supports reduced /
    flexible parameterizations over an arbitrary subset of elementary error generators, with each
    coefficient an independent (unconstrained) real parameter.
    Valid param modes: 'static', 'elements'.
    """
    _coeff_ndim = 1
    _block_dtype = 'd'
    _uses_cache_mx = False
    _superops_matrix_structured = False
    _PARAM_CLASSES = {'static': _StaticParam, 'elements': _EEGVectorElements}

    def _init_labels(self, basis, basis_element_labels, error_generator_labels):
        if error_generator_labels is not None:
            if basis_element_labels is not None:
                raise ValueError("Specify at most one of `basis_element_labels` and "
                                 "`error_generator_labels` for an 'other_unconstrained' block!")
            eeg_labels = tuple(error_generator_labels)
            for lbl in eeg_labels:
                if lbl.errorgen_type not in ('S', 'C', 'A'):
                    raise ValueError("'other_unconstrained' blocks manage only 'S', 'C', and 'A' "
                                     "error generators; got type '%s'." % lbl.errorgen_type)
            self._eeg_labels = eeg_labels
            # distinct basis element labels appearing across the managed eegs, in first-seen order
            # (so that a block built from the full bel set round-trips its basis_element_labels).
            bels = []
            for lbl in eeg_labels:
                for bl in lbl.basis_element_labels:
                    if bl not in bels:
                        bels.append(bl)
            self._bel_labels = tuple(bels)
        else:
            bels = tuple(basis_element_labels) if (basis_element_labels is not None) \
                else tuple(basis.labels[1:])  # Note: don't include identity
            self._bel_labels = bels
            eeg_labels = [_LEEL('S', (P,)) for P in bels]
            for i, P in enumerate(bels):
                for Q in bels[i + 1:]:
                    eeg_labels.append(_LEEL('C', (P, Q)))
                    eeg_labels.append(_LEEL('A', (P, Q)))
            self._eeg_labels = tuple(eeg_labels)

    def _coeff_count(self):
        return len(self._eeg_labels)

    # --- superoperator construction (one superop per elementary error generator) ---

    def _elementary_errorgen_create_fn(self):
        # choose the elementary-errorgen constructor appropriate to this block's basis
        comps = set(self._basis.name.split('*'))
        if comps == {'pp'}:       # normalized Pauli-product basis
            return partial(_lt.create_elementary_errorgen_pauli, normalized_paulis=True)
        elif comps == {'PP'}:     # unnormalized Pauli-product basis
            return _lt.create_elementary_errorgen_pauli
        else:
            return _lt.create_elementary_errorgen

    def _compute_term_generators(self, mxs, d2, sparse):
        # `mxs` is ignored: superops are built per elementary error generator, not per basis element.
        create_fn = self._elementary_errorgen_create_fn()
        basis = self._basis
        nEEG = len(self._eeg_labels)
        superops = [None] * nEEG if sparse else _np.empty((nEEG, d2, d2), 'complex')
        for i, eeg in enumerate(self._eeg_labels):
            bel_mxs = [basis[bl] for bl in eeg.basis_element_labels]
            superops[i] = create_fn(eeg.errorgen_type, *bel_mxs, sparse=sparse)
        return superops

    def create_lindblad_term_superoperators(self, mx_basis='pp', sparse: Union[Literal['auto'], bool]="auto",
                                            include_1norms=False, flat=False):
        assert(self._basis is not None), "Cannot create lindblad superoperators without a basis!"
        basis = self._basis
        sparse = basis.sparse if (sparse == "auto") else sparse
        if len(self._eeg_labels) == 0:
            return ([], []) if include_1norms else []  # short circuit - no superops to return

        sample = basis[self._eeg_labels[0].basis_element_labels[0]]
        d = sample.shape[0]
        d2 = d**2
        mx_basis = _Basis.cast(mx_basis, d2, sparse=sparse)

        # NOTE: cache key is on _eeg_labels (not _bel_labels) so two reduced blocks over the same
        # basis elements but different eeg subsets don't collide in the shared class-level cache.
        cache_key = (self._block_type, tuple(self._eeg_labels), mx_basis, basis)
        if cache_key not in self._superops_cache:
            self._update_superops_cache(None, mx_basis, cache_key)
        cached_superops, cached_superops_1norms = self._superops_cache[cache_key]

        # _superops_matrix_structured is False, so the flat list/array is always what we return.
        if include_1norms:
            return _copy.deepcopy(cached_superops), cached_superops_1norms.copy()
        else:
            return _copy.deepcopy(cached_superops)

    # --- term-object construction (for term-based svterm/cterm evotypes) ---

    def _create_lindblad_term_objects_impl(self, evotype, state_space, mpv, pio):
        Lterms = []

        def O_terms(Um, Un, poly):
            # rank-1 terms for the 'O'-type Lindbladian generator O(Um,Un): rho -> Un rho Um^dag
            #   - 0.5 {Um^dag Un, rho}.  Mirrors the construction in _OtherCoeffBlock.
            Um_dag = Um.conjugate().T
            Un_dag = Un.conjugate().T
            poly_sandwich = 1.0 * poly
            term_sandwich = _term.RankOnePolynomialOpTerm.create_from(
                poly_sandwich, Un, Um, evotype, state_space)

            poly_anti = -0.5 * poly
            op_post = _np.dot(Un_dag, Um)
            term_post = _term.RankOnePolynomialOpTerm.create_from(
                poly_anti, None, op_post, evotype, state_space)

            op_pre = _np.dot(Um_dag, Un)
            term_pre = _term.RankOnePolynomialOpTerm.create_from(
                poly_anti, op_pre, None, evotype, state_space)

            return [term_sandwich, term_post, term_pre]

        def coeff_poly(k, scalar):
            # polynomial multiplying an O-generator for the k-th elementary error generator
            return self._parameterization._coefficient_polynomial(self, k, pio, mpv) * scalar

        for k, eeg in enumerate(self._eeg_labels):
            etype = eeg.errorgen_type
            bels = eeg.basis_element_labels
            if etype == 'S':  # S(P) == O(P,P)
                scaleP, U_P = _mt.to_unitary(self._basis[bels[0]])
                Lterms.extend(O_terms(U_P, U_P, coeff_poly(k, scaleP * scaleP)))
            elif etype == 'C':  # G_C(P,Q) = O(P,Q) + O(Q,P)
                scaleP, U_P = _mt.to_unitary(self._basis[bels[0]])
                scaleQ, U_Q = _mt.to_unitary(self._basis[bels[1]])
                scale = scaleP * scaleQ
                Lterms.extend(O_terms(U_P, U_Q, coeff_poly(k, scale)))
                Lterms.extend(O_terms(U_Q, U_P, coeff_poly(k, scale)))
            elif etype == 'A':  # G_A(P,Q) = 1j*(O(Q,P) - O(P,Q))
                scaleP, U_P = _mt.to_unitary(self._basis[bels[0]])
                scaleQ, U_Q = _mt.to_unitary(self._basis[bels[1]])
                scale = scaleP * scaleQ
                Lterms.extend(O_terms(U_P, U_Q, coeff_poly(k, -1j * scale)))
                Lterms.extend(O_terms(U_Q, U_P, coeff_poly(k, 1j * scale)))
            else:
                raise ValueError("Invalid elementary errorgen type: %s" % etype)
        return Lterms

    # --- elementary-errorgen <-> block_data maps (identity, by construction) ---

    def _elementary_errorgen_indices_impl(self):
        return _collections.OrderedDict((lbl, [(1.0, i)]) for i, lbl in enumerate(self._eeg_labels))

    def _block_data_indices_impl(self):
        return {i: [(1.0, lbl)] for i, lbl in enumerate(self._eeg_labels)}



    # --- overrides that must account for _eeg_labels (vs. the base's _bel_labels) ---

    def convert(self, param_mode):
        return LindbladCoefficientBlock(self._block_type, self._basis,
                                        initial_block_data=self.block_data.copy(),
                                        param_mode=param_mode,
                                        error_generator_labels=self._eeg_labels)

    def is_similar(self, other_coeff_block):
        if not isinstance(other_coeff_block, _OtherUnconstrainedCoeffBlock):
            return False
        return ((self._block_type == other_coeff_block._block_type)
                and (self._param_mode == other_coeff_block._param_mode)
                and (self._eeg_labels == other_coeff_block._eeg_labels)
                and (self._basis == other_coeff_block._basis))

    def _coefficient_labels_impl(self):
        labels = []
        for i, ilbl in enumerate(self._bel_labels):
            for j, jlbl in enumerate(self._bel_labels):
                labels.append("(%s,%s) other error coefficient" % (ilbl, jlbl))
        return labels


class _OtherSparseCoeffBlock(_OtherUnconstrainedCoeffBlock):
    """Selected S coefficients and C/A pair terms with sparse CPTP parameterizations.

    C/A pair terms couple two distinct operator-basis directions. C-type correlation
    coefficients give the real off-diagonal Kossakowski components; A-type active
    coefficients give the negative imaginary components in their ordered label orientation.

    The ``cholesky`` chart requires matching C/A edges, or no A (real factors). Every edge
    endpoint needs an S coefficient. ``contraction`` supports any C/A mask, including zero
    noise and the PSD boundary. ``cholesky completion`` also supports nonchordal and
    unequal component masks, using positive pivots and dependent factor entries, but only
    in the positive-definite interior. ``auto`` selects ``cholesky`` when the mask allows
    it and ``contraction`` otherwise. ``static`` and ``elements`` do not constrain positivity.
    """
    _PARAM_CLASSES = {'static': _StaticParam, 'elements': _EEGVectorElements, 'cholesky': _SparseCholesky,
                      'cholesky completion': _CompletedCholesky, 'contraction': _Contraction,
                      'auto': _SparseCholesky}

    def _init_labels(self, basis, basis_element_labels, error_generator_labels):
        super()._init_labels(basis, basis_element_labels, error_generator_labels)
        _lt._check_basis_factor(basis)
        labels = self._eeg_labels
        if len(set(labels)) != len(labels):
            raise ValueError("Sparse Lindblad blocks require distinct elementary error generators.")
        directions = [lbl.basis_element_labels[0] for lbl in labels if lbl.errorgen_type == 'S'
                      and len(lbl.basis_element_labels) == 1]
        self._bel_labels = tuple(directions)
        pos = {bel: i for i, bel in enumerate(directions)}
        if any(bel not in basis.labels[1:] for bel in directions):
            raise ValueError("Sparse Lindblad directions must be nonidentity elements of the basis.")
        edges = {'C': set(), 'A': set()}
        self._coefficient_pairs = []
        for lbl in labels:
            bels = lbl.basis_element_labels
            if len(bels) != (1 if lbl.errorgen_type == 'S' else 2) or any(b not in pos for b in bels):
                raise ValueError("Every sparse C/A endpoint must have an S coefficient, with valid label arity.")
            i, j = (pos[bels[0]], pos[bels[-1]])
            self._coefficient_pairs.append((i, j))
            if lbl.errorgen_type != 'S':
                edge = tuple(sorted((i, j)))
                if i == j or edge in edges[lbl.errorgen_type]:
                    raise ValueError("Sparse C/A pair terms must have distinct endpoints and no reversed duplicates.")
                edges[lbl.errorgen_type].add(edge)
        n = len(directions)
        pairs = list(edges['C'] | edges['A'])
        rows, cols = _np.array(pairs, dtype=int).reshape(-1, 2).T
        pattern = _sps.coo_array((_np.ones(len(pairs)), (rows, cols)), shape=(n, n))
        if self._param_mode in ('static', 'elements'):
            return
        if self._param_mode == 'contraction':
            self._init_contraction()
            return
        ordering = _sparsechol.perfect_elimination_ordering(pattern)
        if self._param_mode == 'auto':
            self._param_mode = ('cholesky' if ordering is not None and (not edges['A'] or edges['A'] == edges['C'])
                                else 'contraction')
            self._parameterization = self._PARAM_CLASSES[self._param_mode]()
            if self._param_mode == 'contraction':
                self._init_contraction()
                return
        if self._param_mode == 'cholesky completion':
            self.cholesky_structure = _sparsechol.CholeskyStructure(pattern, ordering=self._cholesky_ordering)
            self._init_completion(edges)
            return
        if edges['A'] and edges['A'] != edges['C']:
            raise ValueError("Sparse Cholesky requires matching C/A supports, or no A coefficients.")
        if ordering is None:
            raise ValueError("Sparse Cholesky requires chordal Kossakowski support; this pattern needs fill.")
        if self._cholesky_ordering is not None:
            ordering = self._cholesky_ordering
        self.cholesky_structure = st = _sparsechol.CholeskyStructure(pattern, ordering=ordering)
        if st.factor_pattern.nnz != n + st.pattern.nnz // 2:
            raise ValueError("The requested Cholesky ordering introduces fill; use 'cholesky completion'.")

        # One real diagonal parameter, then real (and optionally imaginary) strict-lower
        # entries in CSC order. These arrays describe a linear map from parameters to L.
        m = len(st.rows)
        complex_factor = bool(edges['A'])
        self._factor_rows = _np.concatenate((_np.arange(n), st.rows, st.rows if complex_factor else [])).astype(int)
        self._factor_cols = _np.concatenate((_np.arange(n), st.cols, st.cols if complex_factor else [])).astype(int)
        self._factor_imag = _np.arange(n + m * (2 if complex_factor else 1)) >= n + m
        self._factor_scales = _np.where(self._factor_imag, 1j, 1.)

        # Expand just the requested coefficients of L L^dag into quadratic monomials.
        # Forward evaluation and derivatives then need neither dense K nor dense dL/dp.
        factor = [{} for _ in range(n)]
        for p, (r, c, scale) in enumerate(zip(self._factor_rows, self._factor_cols, self._factor_scales)):
            factor[st.perm[r]].setdefault(c, []).append((p, scale))
        terms, left, right, scales = [], [], [], []
        for t, (lbl, (i, j)) in enumerate(zip(labels, self._coefficient_pairs)):
            polynomial = {}
            for c in factor[i].keys() & factor[j].keys():
                for p, sp in factor[i][c]:
                    for q, sq in factor[j][c]:
                        z = sp * sq.conjugate()
                        s = -z.imag if lbl.errorgen_type == 'A' else z.real
                        key = tuple(sorted((p, q)))
                        polynomial[key] = polynomial.get(key, 0.) + s
            for (p, q), s in sorted(polynomial.items()):
                if s:
                    terms.append(t); left.append(p); right.append(q); scales.append(s)
        self._quadratic_terms = (_np.array(terms, dtype=int), _np.array(left, dtype=int),
                                 _np.array(right, dtype=int), _np.array(scales, dtype=float))

    def _init_contraction(self):
        n = len(self._bel_labels)
        diagonal, terms = {}, []
        for index, (lbl, (i, j)) in enumerate(zip(self._eeg_labels, self._coefficient_pairs)):
            if lbl.errorgen_type == 'S':
                diagonal[i] = index
            else:
                terms.append((index, i, j, lbl.errorgen_type == 'A'))
        self._contraction_diagonal = _np.array([diagonal[i] for i in range(n)], int)
        k, i, j, active = (_np.array(a, dtype=d) for a, d in zip(zip(*terms) if terms else ([], [], [], []),
                                                                  (int, int, int, bool)))
        self._contraction_terms = (k, i, j, active)
        # One gauge per connected component of the C/A graph, so separate components stay independent.
        graph = _sps.coo_array((_np.ones(len(k)), (i, j)), shape=(n, n))
        count, labels = _csgraph.connected_components(graph, directed=False)
        components = []
        for comp in range(count):
            members = _np.flatnonzero(labels[i] == comp) if len(k) else _np.empty(0, int)
            if not len(members):
                continue
            local = _np.flatnonzero(labels == comp)
            position = {int(d): a for a, d in enumerate(local)}
            li = _np.array([position[int(d)] for d in i[members]], int)
            lj = _np.array([position[int(d)] for d in j[members]], int)
            z = _np.where(active[members], -1j, 1. + 0j)  # A label (i, j): K_ij = -1j * A
            components.append((local, members, li, lj, z))
        self._contraction_components = components

    def _init_completion(self, edges):
        st = self.cholesky_structure
        self._completion_rows = st.factor_pattern.indices.copy()
        self._completion_cols = _np.repeat(_np.arange(st.n), _np.diff(st.factor_pattern.indptr))
        self._completion_diagonal = _np.flatnonzero(self._completion_rows == self._completion_cols)
        row_entries = [{} for _ in range(st.n)]
        for index, (i, j) in enumerate(zip(self._completion_rows, self._completion_cols)):
            row_entries[i][j] = index
        free_real, free_imag, steps = [], [], []
        for index, (i, j) in enumerate(zip(self._completion_rows, self._completion_cols)):
            if i == j:
                continue
            edge = tuple(sorted((int(st.perm[i]), int(st.perm[j]))))
            re, im = edge in edges['C'], edge in edges['A']
            if re:
                free_real.append(index)
            if im:
                free_imag.append(index)
            if not (re and im):
                common = sorted(c for c in row_entries[i].keys() & row_entries[j].keys() if c < j)
                steps.append((index, j, re, im, _np.array([row_entries[i][c] for c in common], int),
                               _np.array([row_entries[j][c] for c in common], int)))
        self._completion_steps = steps
        self._completion_free = _np.array(free_real + free_imag, int)
        self._completion_free_params = st.n + _np.arange(len(self._completion_free))
        self._completion_scales = _np.r_[_np.ones(len(free_real)), 1j * _np.ones(len(free_imag))]
        terms, left, right, coefficient_terms = [], [], [], []
        for index, (i, j) in enumerate(self._coefficient_pairs):
            a, b = row_entries[st.inv[i]], row_entries[st.inv[j]]
            common = sorted(a.keys() & b.keys())
            l, r = _np.array([a[c] for c in common], int), _np.array([b[c] for c in common], int)
            coefficient_terms.append((l, r))
            terms.extend([index] * len(common)); left.extend(l); right.extend(r)
        self._completion_coefficient_terms = coefficient_terms
        self._completion_gram = (_np.array(terms, int), _np.array(left, int), _np.array(right, int))
        self._completion_active = _np.array([lbl.errorgen_type == 'A' for lbl in self._eeg_labels])

    def _set_block_data(self, block_data, truncate):
        if block_data is None and self._param_mode == 'cholesky completion':
            block_data = _np.array([float(lbl.errorgen_type == 'S') for lbl in self._eeg_labels])
        super()._set_block_data(block_data, truncate)

    def _compute_term_generators(self, mxs, d2, sparse):
        if not sparse:
            return super()._compute_term_generators(mxs, d2, sparse)
        # Row-vectorization: vec(A rho B) = (A kron B.T) vec(rho). Construct
        # the sparse superoperators directly, avoiding a loop over all d^2 matrix units
        # for every selected generator. The validated directions are Hermitian.
        directions = {b: _sps.csr_matrix(self._basis[b]) for b in self._bel_labels}
        d = next(iter(directions.values())).shape[0]
        identity = _sps.eye(d, format='csr', dtype=complex)
        kron = lambda a, b: _sps.kron(a, b, format='csr')
        anti = lambda a: kron(a, identity) + kron(identity, a.T)
        superops = []
        last_pair = None
        for label in self._eeg_labels:
            bels = label.basis_element_labels
            p = directions[bels[0]]
            if label.errorgen_type == 'S':
                result = kron(p, p.T) - .5 * anti(p @ p)
            else:
                if bels != last_pair:
                    q = directions[bels[1]]
                    pq, qp = p @ q, q @ p
                    sandwich_pq, sandwich_qp = kron(p, q.T), kron(q, p.T)
                    last_pair = bels
                if label.errorgen_type == 'C':
                    result = sandwich_pq + sandwich_qp - .5 * anti(pq + qp)
                else:
                    result = 1j * (sandwich_pq - sandwich_qp + .5 * anti(pq - qp))
            result.eliminate_zeros()
            superops.append(result.tocsr())
        return superops

    def convert(self, param_mode):
        ordering = self.cholesky_structure.perm if hasattr(self, 'cholesky_structure') else None
        return LindbladCoefficientBlock(self._block_type, self._basis, initial_block_data=self.block_data.copy(),
                                        param_mode=param_mode, error_generator_labels=self._eeg_labels,
                                        cholesky_ordering=ordering, completion_normalization=self._completion_normalization)

    def is_similar(self, other_coeff_block):
        if not super().is_similar(other_coeff_block):
            return False
        if self._param_mode in ('cholesky', 'cholesky completion'):
            return (_np.array_equal(self.cholesky_structure.perm, other_coeff_block.cholesky_structure.perm)
                    and (self._param_mode != 'cholesky completion'
                         or self._completion_normalization == other_coeff_block._completion_normalization))
        return True

    def kossakowski_matrix(self):
        """Return the Hermitian coefficient matrix in ``basis_element_labels`` order.

        Its diagonal contains S coefficients. C-type correlation coefficients add
        equally to the real entries of each labelled pair. For an A-type active
        label (i, j), K[i, j] receives -1j * A and K[j, i] receives +1j * A.
        This dense reconstruction is for inspection and inversion, not forward evaluation.
        """
        K = _np.zeros((len(self._bel_labels), len(self._bel_labels)), complex)
        for lbl, (i, j), value in zip(self._eeg_labels, self._coefficient_pairs, self.block_data):
            z = -1j * value if lbl.errorgen_type == 'A' else value
            K[i, j] += z
            if i != j:
                K[j, i] += z.conjugate()
        return K


LindbladCoefficientBlock._BLOCK_TYPES = {
    'ham': _HamCoeffBlock,
    'other_diagonal': _OtherDiagonalCoeffBlock,
    'other': _OtherCoeffBlock,
    'other_unconstrained': _OtherUnconstrainedCoeffBlock,
    'other_sparse': _OtherSparseCoeffBlock,
}


@lru_cache(maxsize=16)
def cached_diag_indices(n):
    return _np.diag_indices(n)
