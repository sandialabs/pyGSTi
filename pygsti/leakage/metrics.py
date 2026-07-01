#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import scipy.linalg as la

from pygsti.baseobjs.basis import TensorProdBasis, Basis, BuiltinBasis
from pygsti.tools import basistools as pgbt
from pygsti.tools import matrixtools as pgmt
from pygsti.tools import optools as pgot

from pygsti.leakage.core import (
    NOTATION, set_docstring, computational_effect,
    computational_projector
)

BasisLike = Union[Basis, str]


# MARK: Choi-induced metrics


@set_docstring(
"""
Return the rank-1 density in M[H⨂H] induced by op_basis' computational effect.
""" + NOTATION)
def tensorized_teststate_density(op_basis: Basis) -> np.ndarray:
    if not op_basis.implies_leakage_modeling:
        udim = int(np.sqrt(op_basis.dim))
        E = np.eye(udim)
    else:
        E = computational_effect(op_basis)
        if la.norm(E.imag) > 0:
            raise ValueError()
    psi = pgbt.stdmx_to_stdvec(E).ravel()
    psi /= la.norm(psi)
    # In a standard leakage basis we have
    #   |psi> = (|00> + |11> + ... + |dim - n_leak - 1>) / sqrt(dim - n_leak).
    rho_test = np.outer(psi, psi)
    return rho_test


@set_docstring(
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

Let rho be the rank-1 density in M[H⨂H] induced by op_basis' computational effect,
and let I denote the element of S[H] that acts as the identity on M[H].

This function returns a triplet consisting of a pyGSTi Basis object for S[H⨂H],
and the vector representations of X⨂I(rho) and Y⨂I(rho) in that basis.
""" + NOTATION)
def apply_tensorized_to_teststate(op_x: np.ndarray, op_y: np.ndarray, op_basis: BasisLike) -> tuple[TensorProdBasis, np.ndarray, np.ndarray]:
    udim = int(np.sqrt(op_x.shape[0]))
    dim = udim**2
    assert op_x.shape == (dim, dim)
    assert op_y.shape == (dim, dim)

    # We need to construct lifted operators "lift_op_x" and "lift_op_y" that act on the
    # tensor product space M[H]⨂M[H] according to the identities
    #
    #   lift_op_x( sigma ⨂ rho ) = op_x(sigma) ⨂ rho
    #   lift_op_y( sigma ⨂ rho ) = op_y(sigma) ⨂ rho
    #
    # for all sigma, rho in M[H]. The way we do this requires working in the standard (matrix-unit)
    # basis for M[H] and the induced tensor-product basis for M[H]⨂M[H].

    op_basis = Basis.cast(op_basis, dim=dim)
    std_basis = BuiltinBasis('std', dim)
    op_x_std = pgbt.change_basis(op_x, op_basis, std_basis)
    op_y_std = pgbt.change_basis(op_y, op_basis, std_basis)
    idle_gate = np.eye(dim, dtype=np.complex128)
    lift_op_x_std = np.kron(op_x_std, idle_gate)
    lift_op_y_std = np.kron(op_y_std, idle_gate)
    ten_std_basis = TensorProdBasis((std_basis, std_basis))
    # ^ lift_op_x_std and lift_op_y_std are implicitly in ten_std_basis.

    # We compare these lifted operators by how they act on specific state in M[H]⨂M[H].
    rho_test = tensorized_teststate_density(op_basis)
    rho_test_superket = pgbt.stdmx_to_vec(rho_test, ten_std_basis).ravel()
    temp1 = lift_op_x_std @ rho_test_superket
    temp2 = lift_op_y_std @ rho_test_superket

    return ten_std_basis, temp1, temp2


@set_docstring(
"""
Return the (subspace) Choi state of the superoperator X represented by `op_x` in `op_basis`.

Let ρ be the rank-1 density in M[H⨂H] induced by op_basis' computational effect, and let
I ∈ S[H] act as the identity on M[H]. This function returns the density matrix X⨂I(ρ),
expressed in the standard (matrix-unit) basis of H⨂H.

When `op_basis` implies leakage modeling, ρ is the maximally entangled state of the
computational subspace C (against a reference copy of H), so the result is the Choi matrix
of X seen only through inputs supported on C. Otherwise ρ is the maximally entangled state
of all of H and the result is the ordinary Choi state of X.
""" + NOTATION)
def choi_state(op_x: np.ndarray, op_basis: BasisLike) -> np.ndarray:
    dim = round(op_x.size ** 0.5)
    assert op_x.shape == (dim, dim)

    op_basis  = Basis.cast(op_basis, dim=dim)
    std_basis = BuiltinBasis('std', dim)
    op_x_std  = pgbt.change_basis(op_x, op_basis, std_basis)
    idle_gate = np.eye(dim, dtype=np.complex128)

    lift_op_x_std = np.kron(op_x_std, idle_gate)
    ten_std_basis = TensorProdBasis((std_basis, std_basis))

    rho_test_mat  = tensorized_teststate_density(op_basis)
    rho_test_vec  = pgbt.stdmx_to_vec(rho_test_mat, ten_std_basis).ravel()
    choi_superket = lift_op_x_std @ rho_test_vec
    choi_state_mx = pgbt.vec_to_stdmx(choi_superket, ten_std_basis, keep_complex=True)
    return choi_state_mx


CHOI_INDUCED_METRIC_TEMPLATE = \
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

Let rho be the rank-1 density in M[H⨂H] induced by op_basis' computational effect,
and let I denote the element of S[H] that acts as the identity on M[H].

This function returns the %s between X⨂I(rho) and Y⨂I(rho).
""" + NOTATION


@set_docstring(CHOI_INDUCED_METRIC_TEMPLATE % 'entanglement fidelity')
def subspace_entanglement_fidelity(op_x: np.ndarray, op_y: np.ndarray, op_basis) -> float:
    ten_std_basis, temp1, temp2 = apply_tensorized_to_teststate(op_x, op_y, op_basis)
    temp1_mx = pgbt.vec_to_stdmx(temp1, ten_std_basis, keep_complex=True)
    temp2_mx = pgbt.vec_to_stdmx(temp2, ten_std_basis, keep_complex=True)
    ent_fid = pgot.fidelity(temp1_mx, temp2_mx)
    return ent_fid  # type: ignore


@set_docstring(CHOI_INDUCED_METRIC_TEMPLATE % 'Jamiolkowski trace distance')
def subspace_jtracedist(op_x: np.ndarray, op_y: np.ndarray, op_basis) -> float:
    ten_std_basis, temp1, temp2 = apply_tensorized_to_teststate(op_x, op_y, op_basis)
    temp1_mx = pgbt.vec_to_stdmx(temp1, ten_std_basis, keep_complex=True)
    temp2_mx = pgbt.vec_to_stdmx(temp2, ten_std_basis, keep_complex=True)
    j_dist = pgot.tracedist(temp1_mx, temp2_mx)
    return j_dist  # type: ignore


# MARK: projected metrics


PROJECTION_INDUCED_METRIC_TEMPLATE = \
"""
The pair (op_x, op_y) represent some superoperators (X, Y) in S[H], using op_basis.

We return the %s between op_x @ P and op_y @ P, where P is the computational
projector of op_basis.
""" + NOTATION


@set_docstring(PROJECTION_INDUCED_METRIC_TEMPLATE % 'Frobenius distance')
def subspace_superop_fro_dist(op_x: np.ndarray, op_y: np.ndarray, op_basis: Basis) -> float:
    diff = op_x -  op_y
    if op_basis.implies_leakage_modeling:
        P = computational_projector(op_basis) # type: ignore
    else:
        P = pgmt.IdentityOperator()
    return la.norm(diff @ P)  # type: ignore


@set_docstring(PROJECTION_INDUCED_METRIC_TEMPLATE % 'diamond distance')
def subspace_diamonddist(op_x: np.ndarray, op_y: np.ndarray, op_basis) -> float:
    """
    Here we give a brief motivating derivation for defining the subspace diamond norm in
    the way that we have. This derivation won't convince a skeptic that our definition
    is the best-possible.

    Suppose we canonically measure the distance between two superoperators (X, Y) by

        D(X, Y; H) = max || (X - Y) v ||
                            v is in M[H],                   (Eq. 1)
                            tr(v) = 1,
                            v is positive

    for some norm || * ||.
    
    We arrive at a natural analog of this metric when (X, Y) are restricted to M[C]
    simply by replacing "H" in (Eq. 1) with "C". 
    
    Using P to denote the orthogonal projector onto M[C], we claim that

        D(X, Y; C) = D(X P, Y P; H).                (Eq. 2)

    Here's a proof of that claim:
    
    |   It's easy to show that P is a positive trace-non-increasing map. In particular,
    |   if u = P v, then the matrix representations of u and v are
    |
    |      mat(v) = [v11,  v12]         and      mat(u) = [v11,  0]
    |               [v21,  v22]                           [  0,  0],
    |    
    |   where v11 and v22 are psd if v is positive. From here the claim follows once
    |   you've convinced yourself that the pair of problems below have the same optimal
    |   objective value
    |
    |       max || (X - Y) P v ||         and        max || (X - Y) P v || 
    |           mat(v) = [v11, v12]                         mat(v) = [v11, v12]
    |                    [v21, v22]                                  [v21, v22]
    |           mat(v) is PSD                               v11 is PSD
    |           tr(v11) + tr(v22) = 1                       tr(v11) <= 1.

    This can be taken a little further. The proof's argument goes through unchanged if,
    instead of starting with the objective || (X - Y) v ||, we started with f((X - Y) v),
    where f satisfies the property that f(c v) >= f(v) whenever c is a scalar >= 1.
    """
    from pygsti.tools.optools import diamonddist
    if op_basis.implies_leakage_modeling:
        P = computational_projector(op_basis)
    else:
        P = pgmt.IdentityOperator()
    val : float = diamonddist(op_x @ P, op_y @ P, op_basis, return_x=False) / 2 # type: ignore
    return val


# MARK: transport profiles


TRANSPORT_PROFILES_TEMPLATE = \
"""
This function returns a description of how a gate (in some process matrix representation)
transports population from SUB to the orthogonal complement of SUB in H.

Underlying mathematics
----------------------
Our subspace of interest is represented by the unique Hermitian operator E_SUB satisfying
⟨ E_SUB, ρ ⟩ = 1 and ρ = E_SUB ρ E_SUB for all densities ρ ∈ M[SUB]. %s

The process matrix `op` is the mx_basis representation of a CPTP map G : M[H] ➞ M[H].

Consider an experimental protocol where a system is prepared in state ρ ∈ M[SUB], evolved
to G(ρ), and then measured with the 2-element POVM {E_SUB, 1 - E_SUB}. Runs of this protocol
that result in the `1 - E_SUB` measurement outcome are called transport events.

The population transport profile of (G,SUB) is a full specification of the probabilties
of transport events, considering all possible choices of ρ ∈ M[SUB]. This profile is 
encoded in the Hermitian operator

    E_transport := (E_SUB) G^{†}(1 - E_SUB) (E_SUB) ∈  M[SUB] ⊂ M[H].

This is a valid representation, since

    Pr{ G transports ρ | ρ ∈ M[SUB] } 
        = ⟨       1 - E_SUB  , G(ρ) ⟩    // definition of a transport event
        = ⟨ G^{†}(1 - E_SUB) ,   ρ  ⟩    // definition of the adjoint
        = ⟨     E_transport  ,   ρ  ⟩.   // using ρ = E_SUB ρ E_SUB


Function output and interpretation
----------------------------------
This function returns a tuple with eigenvalues and eigenvectors of E_transport.

Say this tuple is `(rates, states)`. If G is CPTP, then

    rates[-1] ≤ Pr{ G transports ρ | ρ ∈ M[SUB] } ≤ rates[0].

The upper bound is G's maximum transport of population (Max TOP) out of SUB, and
is acheived by applying G to the pure state `states[0]` ∈ SUB.

The lower bound is G's minimum transport of population (Min TOP) out of SUB (assuming
population starts in SUB) and is acheived by applying G to `states[-1]` ∈ SUB.

The length of `rates` and `states` is equal to the rank of E_SUB.

"""

@set_docstring(
TRANSPORT_PROFILES_TEMPLATE.replace('SUB', 'sub') % '' + """
Failure modes
-------------
We raise a ValueError if pygsti.tools.is_projector(E_sub, E_sub_tol) returns False.
""" + NOTATION )
def pop_transport_profile(E_sub: np.ndarray, op: np.ndarray, mx_basis: Basis, E_sub_tol=1e-14) -> tuple[np.ndarray,list[np.ndarray]]:
    n = int(np.sqrt(E_sub.size))
    if not pgmt.is_projector(E_sub, E_sub_tol):
        msg = \
        f"""
        The argument E_sub must be an orthogonal projector. The provided value was

            E_sub = {E_sub};

        this failed the tests in pygsti.tools.is_projector at tolerance={E_sub_tol}.
        """
        raise ValueError(msg)

    E_sub_perp_mat  = np.eye(n) - E_sub
    E_sub_perp_vec  = pgbt.stdmx_to_vec(E_sub_perp_mat, mx_basis)
    transport_E_vec  = op.T @ E_sub_perp_vec
    transport_E_mat  = pgbt.vec_to_stdmx(transport_E_vec, mx_basis, keep_complex=True)
    transport_E_mat  = E_sub @ transport_E_mat @ E_sub

    rates, states = la.eigh(transport_E_mat)
    dim_proj = round(np.trace(E_sub).real)
    ind = np.argsort(np.abs(rates))[::-1][:dim_proj]
    rates  = rates[ind]
    states = [s for s in states.T[ind]]

    return rates, states


@set_docstring(
TRANSPORT_PROFILES_TEMPLATE.replace('SUB', 'C') % """
This is the computational\neffect of mx_basis.
""" + NOTATION )
def gate_leakage_profile(op: np.ndarray, mx_basis: Basis) -> tuple[np.ndarray, list[np.ndarray]]:
    mx_basis   = Basis.cast(mx_basis, dim=int(np.sqrt(op.size)))
    E_comp_mat = computational_effect(mx_basis)
    dim_comp   = round(np.trace(E_comp_mat).real)
    if dim_comp**2 == E_comp_mat.size:
        msg = \
        """
        The provided basis suggests that the computational subspace is equal to
        the entire system Hilbert space. Returning with an empty leakage profile!
        """
        warnings.warn(msg)
        return np.empty((0,)), []

    rates, states = pop_transport_profile(E_comp_mat, op, mx_basis)
    return rates, states


@set_docstring(
TRANSPORT_PROFILES_TEMPLATE.replace('SUB', '[C^⟂]') % """This operator is
the identity in M[H] minus the computational effect of mx_basis.
""" + NOTATION )
def gate_seepage_profile(op, mx_basis) -> tuple[np.ndarray, list[np.ndarray]]:
    mx_basis = Basis.cast(mx_basis, dim=int(np.sqrt(op.size)))
    E_comp_mat = computational_effect(mx_basis)
    n = round(E_comp_mat.size ** 0.5)
    E_leak_mat = np.eye(n) - E_comp_mat
    if round(np.trace(E_comp_mat).real) == n:
        msg = \
        """
        The provided basis suggests that the computational subspace is equal to
        the entire system Hilbert space. Returning with an empty seepage profile!
        """
        warnings.warn(msg)
        return np.empty((0,)), []

    rates, states = pop_transport_profile(E_leak_mat, op, mx_basis)
    return rates, states
