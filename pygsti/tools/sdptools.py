"""
Functions for constructing semidefinite programming models
"""
#***************************************************************************************************
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
if TYPE_CHECKING:
    import cvxpy as cp

import numpy as np
import scipy.linalg as la

from pygsti.tools import basistools as bt
from pygsti.tools import jamiolkowski as jam
from pygsti.tools import lindbladtools as lt
from pygsti.tools import matrixtools as mt
from pygsti.baseobjs import basis as pgb
from pygsti.baseobjs.basis import Basis, ExplicitBasis, DirectSumBasis


try:
    import cvxpy as cp
    old_cvxpy = bool(tuple(map(int, cp.__version__.split('.'))) < (1, 0))
    CVXPY_ENABLED = not old_cvxpy
except:
    CVXPY_ENABLED = False


def diamond_norm_model_jamiolkowski(J):
    # return a model for computing the diamond norm.
    #
    # Uses the primal SDP from arXiv:1207.5726v2, Sec 3.2
    #
    # Maximize 1/2 ( < J, X > + < J.dag, X.dag > )
    # Subject to  [[ I otimes rho0,       X        ],
    #              [      X.dag   ,   I otimes rho1]] >> 0
    #              rho0, rho1 are density matrices
    #              X is linear operator
    #
    # ".dag" returns the adjoint.
    dim = J.shape[0]
    smallDim = int(np.sqrt(dim))
    assert dim == smallDim**2

    rho0 = cp.Variable((smallDim, smallDim), name='rho0', hermitian=True)
    rho1 = cp.Variable((smallDim, smallDim), name='rho1', hermitian=True)
    X = cp.Variable((dim, dim), name='X', complex=True)
    Y = cp.real(X)
    Z = cp.imag(X)
    # <J, X>         = J.dag.ravel() @ X.ravel()
    # <J.dag, X.dag> = J.ravel() @ X.dag.ravel() = conj(<J, X>)
    # 
    # ---> real(<J, X>) = 1/2 (<J, X> + <J.dag, X.dag>)
    # ---> can skip the factor 1/2 if we just form real(<J,X>) directly.
    # 

    K = J.real
    L = J.imag
    if hasattr(cp, 'scalar_product'):
        objective_expr = cp.scalar_product(K, Y) + cp.scalar_product(L, Z)
    else:
        Kf = K.flatten(order='F')
        Yf = Y.flatten(order='F')
        Lf = L.flatten(order='F')
        Zf = Z.flatten(order='F')
        objective_expr = Kf @ Yf + Lf @ Zf

    objective = cp.Maximize(objective_expr)

    ident = np.identity(smallDim, 'd')
    kr_tau0 = cp.kron(ident, cp.imag(rho0))
    kr_tau1 = cp.kron(ident, cp.imag(rho1))
    kr_sig0 = cp.kron(ident, cp.real(rho0))
    kr_sig1 = cp.kron(ident, cp.real(rho1))

    block_11 = cp.bmat([[kr_sig0 ,    Y   ],
                         [   Y.T  , kr_sig1]])
    block_21 = cp.bmat([[kr_tau0 ,    Z   ],
                         [   -Z.T , kr_tau1]])
    block_12 = block_21.T
    mat_joint = cp.bmat([[block_11, block_12],
                          [block_21, block_11]])
    constraints = [
        mat_joint >> 0,
        rho0 >> 0,
        rho1 >> 0,
        cp.trace(rho0) == 1.,
        cp.trace(rho1) == 1.
    ]
    prob = cp.Problem(objective, constraints)
    return prob, [X, rho0, rho1]


def diamond_norm_canon(arg : cp.Expression, basis) -> Tuple[cp.Expression, List[cp.Constraint]]:
    """
    This more or less implements canonicalization of the nonlinear expression
    \|arg\|_{\diamond} into CVXPY Constraints and a representation of its epigraph.
    The canonicalization isn't quite "complete" in CVXPY's usual sense, which would
    require that the epigraph is affine and that no structured variables (like
    Hermitian matrices) are used.
    """
    constraints = []
    d = arg.shape[0]
    small_d = int(np.sqrt(d))
    assert d == small_d**2
    assert arg.shape == (d, d)
    Jarg = jam.jamiolkowski_iso(arg, basis, basis, normalized=False)
    Y0 = cp.Variable(shape=(d, d), hermitian=True)
    Y1 = cp.Variable(shape=(d, d), hermitian=True)
    bmat = cp.bmat([
        [ Y0           ,   -Jarg],
        [-Jarg.T.conj(),    Y1  ]
    ])
    constraints.append(bmat >> 0)
    TrX_Y0 = cp.partial_trace(Y0, [small_d, small_d], 0)
    TrX_Y1 = cp.partial_trace(Y1, [small_d, small_d], 0)
    expr0 = cp.lambda_max(TrX_Y0)
    expr1 = cp.lambda_max(TrX_Y1)
    epi = (expr0 + expr1)/2
    return epi, constraints


def cptp_superop_variable(purestate_dim: int, basis) -> Tuple[cp.Expression, List[cp.Constraint]]:
    d = purestate_dim ** 2
    basis = pgb.Basis.cast(basis, d)
    constraints = []
    if basis.first_element_is_identity:
        toprow = np.zeros((1,d))
        toprow[0] = 1
        X_free = cp.Variable((d-1, d))
        X = cp.vstack((toprow, X_free))
    else:
        X = cp.Variable((d, d))
        matI = np.eye(purestate_dim)
        vecI = bt.stdmx_to_vec(matI, basis)
        constraints.append(X.T @ vecI == vecI)
        """
        Let X be the process matrix for a gate "G". We have
                tr(G(rho)) = < I, G(rho) >
                           = < vec(I), vec(G(rho)) >
                           = < vec(I), X @ vec(rho) >
                           = < X.T @ vec(I), vec(rho) >.
        Therefore tr(G(rho)) = tr(rho) for all rho iff X.T @ vec(I) == vec(I).
        """
    J = jam.jamiolkowski_iso(X, basis, basis, normalized=True)
    constraints.append(J >> 0)
    return X, constraints


def diamond_distance_projection_model(superop: np.ndarray, basis: Basis, leakfree=False, seepfree=False, n_leak=0, cptp=True, subspace_diamond=False):
    assert CVXPY_ENABLED
    dim_mixed = superop.shape[0]
    dim_pure = int(np.sqrt(dim_mixed))
    assert dim_pure**2 == dim_mixed
    constraints = []
    if cptp:
        proj_superop, cons = cptp_superop_variable(dim_pure, basis)
        constraints.extend(cons)
    else:
        proj_superop = cp.Variable((dim_mixed, dim_mixed))
    diamondnorm_arg = superop - proj_superop
    if (leakfree or seepfree or subspace_diamond):
        assert n_leak == 1
        from pygsti.tools.optools import leading_dxd_submatrix_basis_vectors
        dim_pure_compsub  = dim_pure - n_leak
        U = leading_dxd_submatrix_basis_vectors(dim_pure_compsub, dim_pure, basis)
        P = U @ U.T.conj()
        I = np.eye(dim_mixed)
        if leakfree:
            constraints.append( (I -  P) @ proj_superop @ U == 0 )
        if seepfree:
            constraints.append( U.T @ proj_superop @ (I - P) == 0 )
        if subspace_diamond:
            diamondnorm_arg = diamondnorm_arg @ P
    expr, cons = diamond_norm_canon(diamondnorm_arg, basis)
    objective = cp.Minimize(expr / 2)
    # ^ We define the diamond distance between two channels as 
    #   1/2 the diamond norm of their difference.
    constraints.extend(cons)
    problem = cp.Problem(objective, constraints)
    viable_solvers = [solver for solver in ['MOSEK', 'CLARABEL', 'CVXOPT'] if solver in cp.installed_solvers()]
    return problem, proj_superop, viable_solvers


def root_fidelity_canon(sigma: cp.Expression, rho: cp.Expression) -> Tuple[cp.Expression, List[cp.Constraint]]:
    """
    pyGSTi defines fidelity as

        F(sigma, rho) = tr([sigma^{1/2} rho sigma^{1/2}]^{1/2})^2.
    
    Others (including Neilson and Chuang, Sect. 9.2.2) define it without the
    square on the trace. We'll call the unsquared version the *root fidelity,*
    and denote it by

        \sqrt{F}(sigma, rho) = (F(sigma, rho))^{1/2}.
    
    The root fidelity is jointly concave (Neilson and Chuang, Exercise 9.19).
    In fact, it admits the following semidefinite programming characterization

        \sqrt{F}(sigma, rho) = Maximize real(tr(X)) 
                               s.t. [[sigma, X],[X.T.conj(), rho]] >> 0

    -- see Section 7.1.3 of Killoran's PhD thesis, "Entanglement quantification
    and quantum benchmarking of optical communication devices."

    This function returns a pair (expr, constraints) where expr is the hypograph
    variable for \sqrt{F}(sigma, rho) and constraints is a list of CVXPY Constraint
    objects used in the semidefinite representation of the hypograph.
    """
    t = cp.Variable()
    d = sigma.shape[0]
    X = cp.Variable(shape=(d, d), complex=True)
    bmat = cp.hermitian_wrap(cp.bmat([
        [ sigma,        X  ],
        [ X.T.conj(),  rho ]
    ]))
    constraints = [
        bmat >> 0,
        cp.trace(cp.real(X)) >= t
    ]
    return t, constraints
