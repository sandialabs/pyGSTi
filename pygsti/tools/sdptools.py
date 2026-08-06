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

import importlib.util
import numpy as np
import warnings

from typing import Optional, Union, List, Tuple, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    import cvxpy as cp
    ExpressionLike = Union[cp.Expression, np.ndarray]

from pygsti.baseobjs.basis import Basis, BasisLike
from pygsti.tools.matrixtools import assert_hermitian
from pygsti.tools.basistools import stdmx_to_vec
from pygsti.tools.jamiolkowski import jamiolkowski_iso
from pygsti.tools.exceptions import CVXPYFailure


CVXPY_ENABLED = importlib.util.find_spec("cvxpy") is not None
_MOSEK_WARNING_PATTERN = ".*Incorrect array format causing data to be copied*"
_CVXPY = None


def _get_cvxpy():
    global _CVXPY
    if _CVXPY is None:
        import cvxpy as cp
        warnings.filterwarnings('ignore', _MOSEK_WARNING_PATTERN)
        _CVXPY = cp
    return _CVXPY


SDP_SOLVER_PRIORITY = ['MOSEK', 'CLARABEL', 'CVXOPT']


def solve_sdp(prob: cp.Problem, **kwargs) -> tuple[np.floating, dict[str, np.ndarray]]:
    cp = _get_cvxpy()

    objective_val : np.floating = np.array(np.nan).item()
    varvals : dict[str, np.ndarray] = dict()
    for i, solver in enumerate(SDP_SOLVER_PRIORITY):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore','.*Solution may be inaccurate.*', UserWarning)
                prob.solve(solver=solver, **kwargs)
            objective_val = prob.value                                       # type: ignore
            varvals.update({k: v.value for (k, v) in prob.var_dict.items()}) # type: ignore
            break
        except (AssertionError, cp.SolverError) as e:
            if solver != 'MOSEK':
                msg = f"Received error {e} when trying to use solver={solver}."
                if i + 1 == len(SDP_SOLVER_PRIORITY):
                    failure_msg  = "Out of solvers. Returning NaN."
                else:
                    failure_msg  = f"Trying {SDP_SOLVER_PRIORITY[i+1]} next."
                msg += f'\n{failure_msg}'
                warnings.warn(msg, CVXPYFailure)

    return objective_val, varvals


def diamond_norm_model_jamiolkowski(J: ExpressionLike) -> tuple[cp.Problem, List[cp.Variable]]:
    # return a model for computing the diamond norm.
    #
    # Uses the primal SDP from arXiv:1207.5726v2, Sec 3.2
    #
    # Throughout comments in this function, "A.dag" is the 
    # Hermitian adjoint (complex conjugate transpose).
    #
    # Maximize 1/2 ( < J, X > + < J.dag, X.dag > )
    # Subject to  [[ I otimes rho0,       X        ],
    #              [      X.dag   ,   I otimes rho1]] >> 0
    #              rho0, rho1 are density matrices
    #              X is linear operator
    #
    cp = _get_cvxpy()
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
    \\|arg\\|_{\\diamond} into CVXPY Constraints and a representation of its epigraph.
    The canonicalization isn't quite "complete" in CVXPY's usual sense, which would
    require that the epigraph is affine and that no structured variables (like
    Hermitian matrices) are used.
    """
    cp = _get_cvxpy()
    constraints = []
    d = arg.shape[0]
    small_d = int(np.sqrt(d))
    assert d == small_d**2
    assert arg.shape == (d, d)
    Jarg = jamiolkowski_iso(arg, basis, basis, normalized=False)
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


def cptp_superop_variable(purestate_dim: int, basis: BasisLike) -> Tuple[cp.Expression, List[cp.Constraint]]:
    cp = _get_cvxpy()
    d = purestate_dim ** 2
    basis = Basis.cast(basis, d)
    constraints = []
    if basis.first_element_is_identity:
        toprow = np.zeros((1,d))
        toprow[0,0] = 1
        X_free = cp.Variable((d-1, d))
        X = cp.vstack((toprow, X_free))
    else:
        X = cp.Variable((d, d))
        matI = np.eye(purestate_dim)
        vecI = stdmx_to_vec(matI, basis)
        constraints.append(X.T @ vecI == vecI)
        """
        Let X be the process matrix for a gate "G". We have
                tr(G(rho)) = < I, G(rho) >
                           = < vec(I), vec(G(rho)) >
                           = < vec(I), X @ vec(rho) >
                           = < X.T @ vec(I), vec(rho) >.
        Therefore tr(G(rho)) = tr(rho) for all rho iff X.T @ vec(I) == vec(I).
        """
    J = jamiolkowski_iso(X, basis, basis, normalized=True)
    constraints.append(J >> 0)
    return X, constraints


def diamond_distance_projection_model(superop: np.ndarray, basis: Basis, leakfree: bool=False, seepfree: bool=False, cptp: bool=True, subspace_diamond: bool=False):
    assert CVXPY_ENABLED
    cp = _get_cvxpy()
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
        assert basis.implies_leakage_modeling
        from pygsti.leakage.core import computational_superkets
        U = computational_superkets(basis)
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


def cp_superop_variable(purestate_dim: int, basis: BasisLike, name: Optional[str] = None) -> Tuple[cp.Variable, List[cp.Constraint]]:
    """
    Return a real CVXPY Variable representing a superoperator (in `basis`),
    together with the constraint that it is completely positive (CP) --
    i.e., that its Choi matrix is positive semidefinite.

    This is the CP-only sibling of :func:`cptp_superop_variable`. Unlike that
    function, no trace-preservation (or trace-nonincreasing) condition is
    imposed here; callers are expected to add whatever trace conditions their
    application requires (see, e.g., :func:`instrument_projection_model`,
    which constrains the *sum* of several such variables to be TP).
    """
    cp = _get_cvxpy()
    d = purestate_dim ** 2
    basis = Basis.cast(basis, d)
    X = cp.Variable((d, d), name=name)
    J = jamiolkowski_iso(X, basis, basis, normalized=True)
    constraints = [J >> 0]
    return X, constraints


INSTRUMENT_PROJECTION_NORMS = ('diamond', 'frobenius', 'spectral')


def instrument_projection_model(member_superops: Sequence[np.ndarray], basis: BasisLike, norm: str = 'frobenius') -> Tuple[cp.Problem, List[cp.Variable]]:
    """
    Return a CVXPY model for projecting the members of a quantum instrument
    onto the set of CPTR (completely positive, trace-nonincreasing) maps
    whose sum is CPTP.

    Given member superoperators G_0, ..., G_{n-1} (e.g., the dense members of
    an :class:`~pygsti.modelmembers.instruments.Instrument` or
    :class:`~pygsti.modelmembers.instruments.TPInstrument`), the model's
    variables X_0, ..., X_{n-1} are constrained so that each X_i is CP and
    sum_i X_i is TP. These constraints suffice for the stated projection:
    the sum of CP maps is CP, and CP members with a TP sum are automatically
    trace-nonincreasing (each member's trace deficit is the sum of the other
    members' nonnegative traces), so no explicit TR constraints are needed.

    The objective is a sum of per-member norms of (X_i - G_i), selected by
    `norm`:

    * 'diamond' : the diamond norm of the superoperator difference, via
      :func:`diamond_norm_canon`. Note this is the *full* diamond norm per
      member (no factor of 1/2 as in the diamond *distance* reported by
      :func:`diamond_distance_projection_model`); the factor would not
      change the minimizer.
    * 'frobenius' (the default) : the *squared* Frobenius norm of the
      superoperator difference, making the objective the true
      (unique-minimizer) Euclidean projection onto the constraint set. The
      squared Frobenius norm is the same whether applied to the superoperator
      or Choi representation.
    * 'spectral' : the spectral norm of the *Choi matrix* of the difference
      (computed with `normalized=True`, i.e. trace-1 Choi matrices for TP
      maps, matching :func:`cptp_superop_variable`; normalization is a
      uniform scaling and does not change the minimizer).

    Parameters
    ----------
    member_superops : sequence of numpy arrays
        The instrument members' superoperator (process) matrices, each of
        shape (D, D) with D a perfect square, expressed in `basis`. These
        must be real (as is the case for, e.g., the Pauli-product basis).

    basis : BasisLike
        The basis in which `member_superops` are expressed.

    norm : {'diamond', 'frobenius', 'spectral'}
        The per-member norm summed in the objective, as described above.

    Returns
    -------
    problem : cp.Problem
        The (unsolved) minimization problem.
    member_vars : list of cp.Variable
        The variables X_i, named 'X0', 'X1', ..., in the same order as
        `member_superops`. After solving, their `.value` attributes (or the
        corresponding entries of :func:`solve_sdp`'s returned dict) hold the
        projected superoperators.
    """
    assert CVXPY_ENABLED
    cp = _get_cvxpy()
    if norm not in INSTRUMENT_PROJECTION_NORMS:
        raise ValueError(f"norm must be one of {INSTRUMENT_PROJECTION_NORMS}, not {norm!r}.")
    n = len(member_superops)
    if n == 0:
        raise ValueError("member_superops must be nonempty.")
    dim_mixed = member_superops[0].shape[0]
    dim_pure = int(np.sqrt(dim_mixed))
    if dim_pure ** 2 != dim_mixed:
        raise ValueError(f"Member dimension {dim_mixed} is not a perfect square.")
    reals = []
    for i, G in enumerate(member_superops):
        if G.shape != (dim_mixed, dim_mixed):
            raise ValueError(f"member_superops[{i}] has shape {G.shape}; expected {(dim_mixed, dim_mixed)}.")
        if np.iscomplexobj(G) and not np.allclose(G.imag, 0.0):
            raise ValueError(f"member_superops[{i}] is not real; this model's variables are real.")
        reals.append(np.real(G))
    basis = Basis.cast(basis, dim_mixed)

    member_vars = []
    constraints = []
    objective_terms = []
    for i, G in enumerate(reals):
        X, cons = cp_superop_variable(dim_pure, basis, name=f'X{i}')
        member_vars.append(X)
        constraints.extend(cons)
        diff = X - G
        if norm == 'diamond':
            epi, cons = diamond_norm_canon(diff, basis)
            constraints.extend(cons)
            objective_terms.append(epi)
        elif norm == 'frobenius':
            objective_terms.append(cp.sum_squares(diff))
        else:  # 'spectral'
            Jdiff = jamiolkowski_iso(diff, basis, basis, normalized=True)
            H = cp.hermitian_wrap(Jdiff)
            # The spectral norm of a Hermitian matrix is max(lambda_max(H), lambda_max(-H));
            # lambda_max(H) alone would miss a dominant negative eigenvalue.
            objective_terms.append(cp.maximum(cp.lambda_max(H), cp.lambda_max(-H)))

    member_sum = sum(member_vars)
    if basis.first_element_is_identity:
        # TP <=> the first row of the process matrix is (1, 0, ..., 0).
        e1 = np.zeros(dim_mixed)
        e1[0] = 1
        constraints.append(member_sum[0, :] == e1)
    else:
        # See the comment in cptp_superop_variable for why this encodes TP.
        matI = np.eye(dim_pure)
        vecI = stdmx_to_vec(matI, basis)
        constraints.append(member_sum.T @ vecI == vecI)

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    return problem, member_vars


def project_instrument_members(member_superops: Sequence[np.ndarray], basis: BasisLike, norm: str = 'frobenius', **solve_kwargs) -> Tuple[List[Optional[np.ndarray]], np.floating]:
    """
    Project the members of a quantum instrument onto the set of CPTR maps
    that sum to a CPTP map, by building the model from
    :func:`instrument_projection_model` and solving it with :func:`solve_sdp`.

    See :func:`instrument_projection_model` for the meanings of
    `member_superops`, `basis`, and `norm`. Any `solve_kwargs` are forwarded
    to the CVXPY solver.

    Returns
    -------
    projected : list of numpy arrays
        The projected member superoperators, in the same order (and basis) as
        `member_superops`. The constraints hold up to solver tolerance
        (typically ~1e-7), so tiny CP/TP violations may remain. If every
        available solver fails, a `CVXPYFailure` warning is emitted (by
        :func:`solve_sdp`) and the entries are None.
    objective_val : float
        The achieved objective value (the sum of per-member norms, per
        `norm`), or NaN on solver failure.
    """
    problem, member_vars = instrument_projection_model(member_superops, basis, norm)
    objective_val, varvals = solve_sdp(problem, **solve_kwargs)
    projected = [varvals.get(X.name(), None) for X in member_vars]
    return projected, objective_val


def root_fidelity_canon(sigma: cp.Expression, rho: cp.Expression) -> Tuple[cp.Expression, List[cp.Constraint]]:
    """
    pyGSTi defines fidelity as

        F(sigma, rho) = tr([sigma^{1/2} rho sigma^{1/2}]^{1/2})^2.
    
    Others (including Neilson and Chuang, Sect. 9.2.2) define it without the
    square on the trace. We'll call the unsquared version the *root fidelity,*
    and denote it by

        \\sqrt{F}(sigma, rho) = (F(sigma, rho))^{1/2}.
    
    The root fidelity is jointly concave (Neilson and Chuang, Exercise 9.19).
    In fact, it admits the following semidefinite programming characterization

        \\sqrt{F}(sigma, rho) = Maximize real(tr(X)) 
                               s.t. [[sigma, X],[X.T.conj(), rho]] >> 0

    -- see Section 7.1.3 of Killoran's PhD thesis, "Entanglement quantification
    and quantum benchmarking of optical communication devices."

    This function returns a pair (expr, constraints) where expr is the hypograph
    variable for \\sqrt{F}(sigma, rho) and constraints is a list of CVXPY Constraint
    objects used in the semidefinite representation of the hypograph.
    """
    cp = _get_cvxpy()
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
