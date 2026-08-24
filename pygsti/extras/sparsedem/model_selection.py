"""
Sparse model selection for DEMs via a non-negative weighted lasso.

Given a dense point estimate of event probabilities and its covariance matrix
(from ``estimation.estimate_dem_and_covariance``), select a sparse event set by
solving

    min_p  0.5 * (p - p_hat)^T Sigma^+ (p - p_hat) + lam * 1^T p,   p >= 0

where Sigma^+ is an eigenvalue-truncated pseudo-inverse (Sigma is singular due
to the multinomial constraint on the observed bitstring probabilities). On the
positive orthant the L1 penalty is linear, so the problem is a smooth QP solved
by projected gradient descent. The penalty strength is chosen automatically by
scoring supports along a decreasing lambda path with BIC, and the winning
support is refit unpenalized (relaxed lasso) with ``estimation.fit_specified_dem``
to remove shrinkage bias.

Like the other dense methods, this works with full 2^n vectors and is only
viable for small detector counts.
"""

import numpy as np
import stim

from .estimation import estimate_dem_and_covariance, fit_specified_dem


def _pinv_sqrt(covariance_matrix: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
    """
    Compute the pseudo-inverse square root of a covariance matrix.

    Eigenvalues below max(eigenvalue) * rcond are truncated (mapped to zero
    weight), projecting out near-null directions such as the multinomial
    constraint. This restricts the Gaussian quadratic form to the subspace
    where the estimate actually varies; flooring instead would give those
    directions an enormous weight that dominates the whitened problem and
    makes the design columns nearly collinear.

    Parameters:
        covariance_matrix: np.ndarray of shape (n, n)
            Symmetric positive semi-definite matrix.
        rcond: float
            Relative eigenvalue cutoff for truncation.

    Returns:
        np.ndarray of shape (n, n)
            Symmetric matrix W with W @ W approximating pinv(covariance_matrix).
    """
    symmetrized = (covariance_matrix + covariance_matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
    if eigenvalues.max() <= 0:
        raise ValueError("Covariance matrix has no positive eigenvalue.")
    keep = eigenvalues > eigenvalues.max() * rcond
    inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, eigenvalues, 1.0)), 0.0)
    return eigenvectors @ np.diag(inv_sqrt) @ eigenvectors.T


def _nn_lasso(
    Q: np.ndarray,
    c: np.ndarray,
    lam: float,
    x0: np.ndarray,
    lipschitz: float,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Solve min_x 0.5 x^T Q x - c^T x + lam * 1^T x subject to x >= 0.

    Projected gradient descent; on the positive orthant the L1 penalty is
    linear so the prox step is a clipped gradient step, producing exact zeros.

    Parameters:
        Q: np.ndarray of shape (m, m)
            Gram matrix A^T A of the whitened design.
        c: np.ndarray of shape (m,)
            Correlation vector A^T y.
        lam: float
            L1 penalty strength (0 for an unpenalized non-negative fit).
        x0: np.ndarray of shape (m,)
            Starting point (warm start).
        lipschitz: float
            Largest eigenvalue of Q.
        tol: float
            Relative convergence tolerance on the iterate.
        max_iter: int
            Maximum number of iterations.

    Returns:
        np.ndarray of shape (m,)
            The solution, with exact zeros off the support.
    """
    step = 1.0 / max(lipschitz, 1e-12)
    x = x0.copy()
    for _ in range(max_iter):
        prev = x
        x = np.maximum(0.0, x - step * (Q @ x - c + lam))
        if np.linalg.norm(x - prev) <= tol * max(1.0, np.linalg.norm(prev)):
            break
    return x


def lasso_select_events(
    estimated_probabilities: np.ndarray,
    covariance_matrix: np.ndarray,
    n_shots: int,
    lam: float = None,
    n_lambdas: int = 50,
    lambda_min_ratio: float = 1e-3,
    rcond: float = 1e-8,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, dict]:
    """
    Select a sparse set of DEM events with a non-negative whitened lasso.

    Whitens the point estimate with the pseudo-inverse square root of the
    covariance and solves the non-negative lasso along a decreasing lambda
    path, scoring each distinct support by BIC on an unpenalized non-negative
    refit (relaxed lasso). Index 0 (the empty mask) is excluded from the
    design.

    Parameters:
        estimated_probabilities: np.ndarray of shape (2**n,)
            Dense event probability estimate, indexed by integer bitmask.
        covariance_matrix: np.ndarray of shape (2**n, 2**n)
            Covariance of the estimate.
        n_shots: int
            Total number of samples (BIC sample size).
        lam: float, optional
            Fixed penalty strength; skips the path and BIC selection.
        n_lambdas: int
            Number of points on the lambda path.
        lambda_min_ratio: float
            Smallest lambda as a fraction of lambda_max.
        rcond: float
            Relative eigenvalue cutoff for whitening.
        tol: float
            Solver convergence tolerance.
        max_iter: int
            Solver iteration cap.

    Returns:
        masks: np.ndarray
            Sorted integer bitmasks of the selected events.
        info: dict
            Path diagnostics: 'lambdas', 'bics', 'supports', 'lambda_best',
            'solution' (the lasso solution over masks 1..2**n-1).
    """
    whitener = _pinv_sqrt(covariance_matrix, rcond=rcond)
    y = whitener @ estimated_probabilities
    A = whitener[:, 1:]  # drop the empty mask; coordinate i <-> mask i + 1

    Q = A.T @ A
    c = A.T @ y
    n_coords = Q.shape[0]
    lipschitz = np.linalg.eigvalsh(Q)[-1]

    info = {
        "lambdas": np.array([]),
        "bics": np.array([]),
        "supports": [],
        "lambda_best": None,
        "solution": np.zeros(n_coords),
    }

    if lam is not None:
        x = _nn_lasso(Q, c, lam, np.zeros(n_coords), lipschitz, tol=tol, max_iter=max_iter)
        info["lambdas"] = np.array([lam])
        info["lambda_best"] = lam
        info["solution"] = x
        info["supports"] = [tuple(np.flatnonzero(x))]
        return np.flatnonzero(x) + 1, info

    lam_max = c.max()
    if lam_max <= 0:
        return np.array([], dtype=int), info

    lambdas = lam_max * np.geomspace(1.0, lambda_min_ratio, n_lambdas)
    info["lambdas"] = lambdas

    def restricted_bic(support):
        # Unpenalized non-negative refit on the support, then BIC = RSS + k log n.
        S = np.array(support)
        Q_SS = Q[np.ix_(S, S)]
        z = _nn_lasso(
            Q_SS, c[S], 0.0, x[S], np.linalg.eigvalsh(Q_SS)[-1], tol=tol, max_iter=max_iter
        )
        rss = y @ y - 2 * c[S] @ z + z @ Q_SS @ z
        return rss + len(S) * np.log(n_shots)

    bic_empty = y @ y
    best_bic, best_support = bic_empty, ()
    scored = {(): bic_empty}
    bics, supports = [], []
    x = np.zeros(n_coords)
    for lam_k in lambdas:
        x = _nn_lasso(Q, c, lam_k, x, lipschitz, tol=tol, max_iter=max_iter)
        support = tuple(np.flatnonzero(x))
        if support not in scored:
            scored[support] = restricted_bic(support)
            supports.append(support)
            bics.append(scored[support])
            if scored[support] < best_bic:
                best_bic, best_support = scored[support], support
                info["lambda_best"] = lam_k
                info["solution"] = x.copy()

    info["bics"] = np.array(bics)
    info["supports"] = supports
    return np.array(best_support, dtype=int) + 1, info


def lasso_dem_selection(
    syndrome_counts: dict,
    lam: float = None,
    n_lambdas: int = 50,
    lambda_min_ratio: float = 1e-3,
    rcond: float = 1e-8,
    atol: float = 1e-4,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    return_path: bool = False,
):
    """
    Estimate a sparse DEM from syndrome counts via non-negative lasso selection.

    Runs the dense estimation with covariance, selects a sparse support with
    ``lasso_select_events``, then refits the selected masks unpenalized with
    ``fit_specified_dem`` to obtain debiased probabilities and a covariance for
    the sparse model. Dense 2^n method — only viable for small detector counts.

    Parameters:
        syndrome_counts: dict
            Mapping bitstrings (e.g., '0011') to counts.
        lam: float, optional
            Fixed penalty strength; skips the BIC path.
        n_lambdas: int
            Number of points on the lambda path.
        lambda_min_ratio: float
            Smallest lambda as a fraction of lambda_max.
        rcond: float
            Relative eigenvalue cutoff for whitening.
        atol: float
            Threshold for zeroing small probabilities in the final DEM.
        tol: float
            Solver convergence tolerance.
        max_iter: int
            Solver iteration cap.
        return_path: bool
            Also return the lambda path diagnostics dict.

    Returns:
        (dem, dem_masks, event_probs, covariance), plus info if return_path.
        If no events are selected, returns an empty DEM with empty arrays.
    """
    estimated_probabilities, covariance_matrix = estimate_dem_and_covariance(syndrome_counts)
    n_shots = sum(syndrome_counts.values())

    masks, info = lasso_select_events(
        estimated_probabilities,
        covariance_matrix,
        n_shots,
        lam=lam,
        n_lambdas=n_lambdas,
        lambda_min_ratio=lambda_min_ratio,
        rcond=rcond,
        tol=tol,
        max_iter=max_iter,
    )

    if len(masks) == 0:
        result = (stim.DetectorErrorModel(), np.array([], dtype=int), np.zeros(0), np.zeros((0, 0)))
    else:
        result = fit_specified_dem(syndrome_counts, masks, atol=atol, return_covariance=True)

    if return_path:
        return (*result, info)
    return result
