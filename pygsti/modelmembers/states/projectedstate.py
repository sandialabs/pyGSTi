"""
A State that holds an internal TPState, and wraps that by projecting
as needed in from_vector(). We implement to_vector() by extracting
the projected density matrix and then returning
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any, Tuple

if TYPE_CHECKING:
    import cvxpy as cp

import numpy as np
import jax
import jax.numpy as jnp
from pygsti.baseobjs.basis import ExplicitBasis, Basis


def strict_triu_op_data(N: int):
    sT = N*(N-1)//2
    row_idx, col_idx = jnp.triu_indices(N, k=1)
    P_rows  = N * row_idx + col_idx
    P_cols  = np.arange(sT)
    return (P_rows, P_cols), (N**2, sT)


def jax_vec_to_strict_upper_tri(vec):
    ell = vec.shape[0]
    n = round( ((8 * ell + 1) ** 0.5 + 1) / 2 )
    if not (n * (n - 1) // 2 == ell):
        raise ValueError("The size of the vector must be a triangular number.")
    coords, shape = strict_triu_op_data(n)
    P_dense = jnp.zeros(shape)
    P_dense = P_dense.at[coords].set(1.0)
    out = (P_dense @ vec).reshape((n, n), order='F').T
    return out


def tpvec_to_herm_parts(vec: cp.Expression | jnp.ndarray):
    import cvxpy as cp
    N2m1 = vec.shape[0]
    N = round((N2m1 + 1)**0.5)
    block0 = vec[ : N*(N-1)//2             ]
    block1 = vec[   N*(N-1)//2 : N*(N-1)   ]
    block2 = vec[                N*(N-1) : ]
    if isinstance(vec, np.ndarray):
        vec = jnp.array(vec)

    if isinstance(vec, cp.Expression):
        tri_real  = cp.vec_to_upper_tri(block0, strict=True)
        tri_imag  = cp.vec_to_upper_tri(block1, strict=True)
        diag      = cp.diag(cp.concatenate([block2, 1 - cp.sum(block2)]))
    elif isinstance(vec, jnp.ndarray):
        tri_real = jax_vec_to_strict_upper_tri(block0)
        tri_imag = jax_vec_to_strict_upper_tri(block1)
        block3   = jnp.atleast_1d( 1 - jnp.sum(block2, axis=0) ) # type: ignore
        diag     = jnp.diag(
            jnp.concatenate([block2, block3]) # type: ignore
        )
    else:
        raise ValueError()

    herm_real = tri_real + tri_real.T + diag
    herm_imag = tri_imag - tri_imag.T
    return herm_real, herm_imag


def herm_parts_to_tpvec(herm_parts: tuple[jnp.ndarray, jnp.ndarray] | tuple[Any, Any], full_superket=False):
    herm_real, herm_imag = herm_parts
    n = herm_imag.shape[0]
    triui = np.triu_indices(n, k=1)
    flat = (n*(n-1)//2,) + herm_imag.shape[2:]
    tri_imag = herm_imag[triui].reshape(flat)
    tri_real = herm_real[triui].reshape(flat)
    num_diag = n if full_superket else n - 1
    diag     = herm_real[np.diag_indices(num_diag)]
    if isinstance(herm_real, jnp.ndarray):
        return jnp.concatenate([tri_real, tri_imag, diag])
    else:
        return np.concatenate([tri_real, tri_imag, diag])
    

def herm_parts_basis(N: int) -> ExplicitBasis:
    triui_rows, triu_cols = np.triu_indices(N, k=1)
    sT = N*(N-1)//2
    mats   = []
    labels = []
    for k in range(N**2):
        mat = np.zeros((N, N), dtype=complex)
        if k < sT:
            # upper triangle, real, read in row-major
            r, c = triui_rows[k], triu_cols[k]
            label = '[ReH_%s]' % str({r,c})
            mat[r, c] = 1.0
            mat[c, r] = 1.0
        elif k < 2*sT:
            # upper triangle, skew, read in row-major
            r, c = triui_rows[k-sT], triu_cols[k-sT]
            label = '[ImH_%s]' % str({r,c})
            mat[r, c] =  1.0j
            mat[c, r] = -1.0j
        else:
            i = k - 2*sT
            label = '[Diag_{%s}]' % i
            mat[i, i] = 1.0
        mats.append(mat)
        labels.append(label)
    name = f'Hermitian matrix-unit basis of {N**2}-dimensional Hilbert-Schmidt space'
    B = ExplicitBasis(mats, labels, name)
    return B


def eigh_via_symmetricification(herm_parts: tuple[jnp.ndarray, jnp.ndarray] | tuple[Any, Any]) -> tuple[ jnp.ndarray | Any , tuple[Any, Any]]:
    """  
    Let A be real-symmetric and B be real skew-symmetric, both of order n.
    If c is a scalar and x and y are real n-vectors satisfying

        c [x] = [ A   B  ][x]
          [y] = [ B'  A  ][y].

    then x - 1j * y is an eigenvector of A + 1j * B with eigenvalue c.

    This function returns a complete set of eigenvalues and eigenvectors
    for the matrix herm_parts[0] + 1j * herm_parts[1], where the 
    eigenvectors are returned in separate matrices containing their real
    and imaginary parts.

    The implementation assumes that `eigh` returns eigenpairs ordered by
    eigenvalue (either increasing or decreasing is fine).
    """
    herm_real, herm_imag = herm_parts
    n = herm_imag.shape[0]
    if isinstance(herm_real, jnp.ndarray):
        sym = jnp.block([[herm_real, herm_imag],[herm_imag.T, herm_real]])
        sevals, sevecs = jnp.linalg.eigh(sym)
    else:
        sym = np.block([[herm_real, herm_imag],[herm_imag.T, herm_real]])
        sevals, sevecs = np.linalg.eigh(sym)
    evals      =  sevals[::2]
    evecs_real =  sevecs[ :n,  ::2 ]
    evecs_imag = -sevecs[  n:, ::2 ]
    return evals, (evecs_real, evecs_imag)


def recollect_expanded_eigendecomposition(evals, evecs_real, evecs_imag):
    """ 
    evals, evecs_real, and evecs_imag are all real-valued.

    Define D = diag(evals) and U = evecs_real + 1j * evecs_imag.
    
    This function returns matrices herm_real and herm_imag where

        herm_real + 1j * herm_imag = U @ D @ U.T.conj().
    """
    temp_real = evals[:, None] * evecs_real.T
    temp_imag = evals[:, None] * evecs_imag.T
    herm_real = evecs_real @ temp_real + evecs_imag @ temp_imag
    herm_imag = evecs_imag @ temp_real - evecs_real @ temp_imag
    return herm_real, herm_imag


def density_projection_model(N: int, param_name='parameters', var_name='projection'):
    import cvxpy as cp
    param  = cp.Parameter(shape=(N**2 - 1,), name=param_name)
    herm_real, herm_imag = tpvec_to_herm_parts(param)
    herm_expr = herm_real + 1j * herm_imag

    X = cp.Variable(shape=(N, N), hermitian=True, name=var_name)
    F = cp.sum_squares(herm_expr - X)
    C = [ 0 << X, cp.trace(cp.real(X)) == 1 ]
    problem = cp.Problem( cp.Minimize(F), C )

    """
    To map V := X.value back to `param` space,
    set triui = np.triu_indices_from(V, k=1),
    then concatenate ...
        V.real[triui],
        V.imag[triui], and
        np.diag(V.real)[:-1].
    """
    return problem


def project_onto_simplex(vec: jnp.ndarray) -> jnp.ndarray:
    # Sort in descending order
    u = jnp.sort(vec)[::-1]
    css = jnp.cumsum(u, axis=0)
    # Find the largest index rho such that
    #   u[rho] > (css[rho] - 1) / (rho + 1)
    inds = jnp.arange(vec.shape[0]) + 1
    cond = u - (css - 1) / inds > 0
    if not jnp.any(cond):
        # if no positive entries, project uniformly on first coordinate
        theta = (css[0] - 1) / 1
    else:
        rho = jnp.where(cond)[0][-1]
        theta = (css[rho] - 1) / (rho + 1)
    proj = jnp.maximum(vec - theta, 0)
    return proj


def project_onto_densities(herm_mx: jnp.ndarray) -> jnp.ndarray:
    v, u = jnp.linalg.eigh(herm_mx)
    udag = u.T.conj()
    proj_v = project_onto_simplex(v)
    density_mx = u @ (proj_v[:, None] * udag)
    return density_mx


def project_onto_densities_realification(herm_parts: jnp.ndarray) -> jnp.ndarray:
    """ 
    Example usage:

        jac = jax.jacobian(project_onto_densities_realification)

        herm_mx = np.random.randn(3,3) + 1j * np.random.randn(3,3)
        herm_mx += temp.T.conj()

        herm_mx_reif = jnp.stack((herm_mx.real, temp.imag))
        # ^ shape (2, 3, 3)
        out = jac(herm_mx_reif)
        # ^ shape (2, 3, 3, 2, 3, 3)

    """
    evals, evecs = eigh_via_symmetricification(herm_parts)  # type: ignore
    proj_evals   = project_onto_simplex(evals)
    density_mx   = recollect_expanded_eigendecomposition(proj_evals, *evecs)
    density_mx   = jnp.stack(density_mx)
    return density_mx


def project_onto_densities_tpvec( tpvec, full_superket=False ):
    herm_parts = tpvec_to_herm_parts( tpvec )
    herm_mx    = jnp.stack(herm_parts)  # type: ignore
    density_mx = project_onto_densities_realification(herm_mx)
    proj_parts = (density_mx[0], density_mx[1])
    tpvec_proj = herm_parts_to_tpvec(proj_parts, full_superket)
    return tpvec_proj



class ProjectedState_CVXPY:

    def __init__(self, N: int) -> None:
        problem = density_projection_model(N)
        self._cvxpy_model = problem
        self._cvxpy_param = problem.param_dict['parameters']
        return

from pygsti.modelmembers import ModelMember
from pygsti.modelmembers.states import State, TPState
from pygsti.tools.basistools import vec_to_stdmx, change_basis


# TODO: need a routine for changing basis of to_vector()/from_vector() with TPState
# versus ProjectedState.





class ProjectedState(TPState):

    def __init__(self, vec: np.ndarray, basis: Basis, evotype="default", state_space=None):
        super().__init__(vec, basis, evotype, state_space)
        self._working_basis = herm_parts_basis(self.state_space.udim)

        self._n2w_np = self._basis.create_transform_matrix(self._working_basis).real
        self._w2n_np = self._working_basis.create_transform_matrix(self._basis).real
        self._n2w = jnp.array( self._n2w_np )
        self._w2n = jnp.array( self._w2n_np )

        self._natural_firstEl = np.array([self.basis.elsize ** -0.25])
        return

    def to_vector(self) -> np.ndarray:
        return super().to_vector()
    
    def _natural_vec_to_working_vec(self, natvec):
        if isinstance(natvec, np.ndarray):
            workvec = self._n2w_np[:-1, :] @  np.concatenate([self._natural_firstEl, natvec])
        else:
            workvec =    self._n2w[:-1, :] @ jnp.concatenate([self._natural_firstEl, natvec])
        return workvec
    
    def _working_vec_to_natural_vec(self, workvec):
        if isinstance(workvec, np.ndarray):
            tail = np.atleast_1d(1 - np.sum(workvec[-(self.state_space.udim - 1):]))
            natvec = self._w2n_np[1:, :] @ np.concatenate([workvec, tail])
        else:
            tail = jnp.atleast_1d(1 - jnp.sum(workvec[-(self.state_space.udim - 1):]))
            natvec = self._w2n[1:, :] @ jnp.concatenate([workvec, tail])
        return natvec
    
    def from_vector(self, v, close=False, dirty_value=True) -> None:
        v = jnp.array(v)
        tpvec_before = self._natural_vec_to_working_vec(v)
        tpvec_after  = project_onto_densities_tpvec( tpvec_before )
        v_projected  = self._working_vec_to_natural_vec( tpvec_after )
        v_projected  = np.asarray(v_projected).astype(np.double)
        super().from_vector(v_projected, False, True)
        return
    
    def set_dense(self, vec):
        super().set_dense(vec)
        self.from_vector(self.to_vector(), False, True) 
        return
    
    def deriv_wrt_params(self, wrt_filter=None):
        assert wrt_filter is None

        def projected_tpvec_onto_densities_as_superket( tpvec ):
            todense_working = project_onto_densities_tpvec( tpvec, full_superket=True )
            todense_natural = self._w2n @ todense_working
            return todense_natural

        jac_fun = jax.jacobian(projected_tpvec_onto_densities_as_superket)
        natural_sket  = jnp.array(self.to_dense('HilbertSchmidt'))
        current_tpvec = self._n2w[:-1, :] @ natural_sket
        jac_val = jac_fun( current_tpvec )
        jac_val = jac_val @ self._n2w[:-1, :]
        # ^ Jacobian mapping natural_sket_in to natural_sket_out.
        jac_val = jac_val[:, 1:]
        # ^ Jacobian mapping the last `from_vector` arg `v` to self.to_dense().

        jac_val = np.asarray( jac_val, dtype=np.double )
        return jac_val
    
    def stateless_data(self) -> Tuple[int]:
        raise NotImplementedError()
    
    @staticmethod
    def torch_base(sd, t_param):
        raise NotImplementedError()
    
    def has_nonzero_hessian(self):
        return False


if __name__ == '__main__':
    temp = np.random.randn(3,3) + 1j * np.random.randn(3,3)
    temp += temp.T.conj()
    tpvec = np.random.rand(3)
    jac2 = jax.jacobian(project_onto_densities_tpvec)
    x,y = tpvec_to_herm_parts(jnp.array(tpvec))
    print()
