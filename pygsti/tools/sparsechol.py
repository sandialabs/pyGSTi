"""
Symbolic tools for sparse Cholesky factorizations: fill-reducing orderings, elimination trees,
and the sparsity patterns of Cholesky factors.

Every function here takes a symmetric sparsity pattern: a square matrix (SciPy sparse or dense)
whose nonzero off-diagonal entries mark the pattern. Numerical values and the diagonal are ignored.
Orderings are permutation arrays `perm` in which `perm[k]` is the index of the k-th eliminated row,
so the matrix being factored is `A[perm][:, perm]`.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import heapq as _heapq
import importlib.util as _importlib_util
import warnings as _warnings

import networkx as _nx
import numpy as _np
import scipy.sparse as _sps

__all__ = ['CholeskyStructure', 'fill_reducing_ordering', 'perfect_elimination_ordering',
           'elimination_tree', 'symbolic_cholesky']

# (backend, module whose absence makes the backend unavailable), in order of preference
_BACKENDS = (('cholmod', 'sksparse'), ('qdldl', 'qdldl'), ('networkx', 'networkx'))


def _adjacency(pattern) -> _sps.csr_matrix:
    """ The symmetrized off-diagonal pattern of `pattern`, as a boolean CSR matrix. """
    A = _sps.csr_matrix(pattern)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A sparsity pattern must be square, not %s." % str(A.shape))
    A = (A != 0).astype(_np.int8)
    A = A + A.T
    A = _sps.triu(A, k=1) + _sps.tril(A, k=-1)
    return (A != 0).tocsr()


def _graph(adj) -> _nx.Graph:
    G = _nx.Graph()
    G.add_nodes_from(range(adj.shape[0]))
    G.add_edges_from(zip(*_sps.triu(adj, k=1).nonzero()))
    return G


def _synthetic_spd(adj) -> _sps.csc_matrix:
    """ A strictly diagonally dominant (hence positive definite) matrix with pattern `adj`. """
    n = adj.shape[0]
    return (adj.astype(float) + _sps.identity(n) * (n + 1)).tocsc()


def _check_perm(perm, n) -> _np.ndarray:
    perm = _np.asarray(perm, dtype=int)
    if perm.shape != (n,) or not _np.array_equal(_np.sort(perm), _np.arange(n)):
        raise ValueError("`perm` must be a permutation of range(%d)." % n)
    return perm


def _ordering_cholmod(adj):
    from sksparse import cholmod
    A = _synthetic_spd(adj)
    if hasattr(cholmod, 'CholeskyFactor'):  # scikit-sparse >= 0.5; the constructor only does symbolic analysis
        return _np.asarray(cholmod.CholeskyFactor(A, order='amd').perm, dtype=int)
    return _np.asarray(cholmod.analyze(A, ordering_method='amd').P(), dtype=int)


def _ordering_qdldl(adj):
    import qdldl
    return _np.asarray(qdldl.Solver(_synthetic_spd(adj)).factors()[2], dtype=int)


def _mcs_ordering(G) -> _np.ndarray:
    """
    The reverse of a maximum cardinality search visit order, which is a perfect elimination
    ordering whenever G is chordal (Tarjan & Yannakakis, 1984).
    """
    weight = {v: 0 for v in G}
    heap = [(0, v) for v in G]  # (-weight, vertex), with stale entries skipped lazily
    visited, order = set(), []
    while heap:
        negw, v = _heapq.heappop(heap)
        if v in visited or -negw != weight[v]:
            continue
        visited.add(v)
        order.append(v)
        for u in G[v]:
            if u not in visited:
                weight[u] += 1
                _heapq.heappush(heap, (-weight[u], u))
    return _np.array(order[::-1], dtype=int)


def _ordering_networkx(adj):
    # Eliminate along a perfect elimination ordering of networkx's minimal (MCS-M) triangulation,
    # so the fill is exactly the triangulation's added edges, and zero on chordal patterns.
    H, _ = _nx.complete_to_chordal_graph(_graph(adj))
    return _mcs_ordering(H)


def fill_reducing_ordering(pattern, *, _backend=None) -> _np.ndarray:
    """
    A fill-reducing elimination ordering of a symmetric sparsity pattern.

    Uses the approximate minimum degree (AMD) ordering from scikit-sparse's CHOLMOD interface if it
    is installed, otherwise from `qdldl` if it is installed; an installed backend that fails (e.g. an
    incompatible version) issues a `RuntimeWarning` and is skipped. Otherwise it falls back to a perfect
    elimination ordering of the minimal (MCS-M) triangulation computed by networkx, so the fill is
    exactly the triangulation's added edges. The fallback is slower on large patterns, but it
    produces no fill on chordal patterns, where AMD may. Only the pattern is used.

    Parameters
    ----------
    pattern : scipy.sparse matrix or numpy.ndarray
        A square matrix whose nonzero off-diagonal entries define the (symmetrized) pattern.

    Returns
    -------
    numpy.ndarray
        A permutation `perm` such that `A[perm][:, perm]` is the matrix to factor.
    """
    adj = _adjacency(pattern)
    n = adj.shape[0]
    if n == 0:
        return _np.zeros(0, dtype=int)
    orderings = {'cholmod': _ordering_cholmod, 'qdldl': _ordering_qdldl, 'networkx': _ordering_networkx}
    if _backend is not None:
        return _check_perm(orderings[_backend](adj), n)
    for backend, module in _BACKENDS[:-1]:
        if _importlib_util.find_spec(module) is None:
            continue  # not installed
        try:
            return _check_perm(orderings[backend](adj), n)
        except Exception as e:  # installed but incompatible, e.g. a changed API
            _warnings.warn("The installed '%s' could not compute an ordering (%s: %s); falling back to the next "
                           "backend." % (module, type(e).__name__, e), RuntimeWarning)
    return _check_perm(_ordering_networkx(adj), n)


def perfect_elimination_ordering(pattern) -> _np.ndarray | None:
    """
    An elimination ordering with no fill, or None if `pattern` is not chordal.

    A symmetric pattern admits an ordering with no Cholesky fill if and only if its graph is
    chordal. Such an ordering is a perfect elimination ordering; other orderings of a chordal
    graph can still produce fill.

    Parameters
    ----------
    pattern : scipy.sparse matrix or numpy.ndarray
        A square matrix whose nonzero off-diagonal entries define the (symmetrized) pattern.

    Returns
    -------
    numpy.ndarray or None
    """
    adj = _adjacency(pattern)
    if adj.shape[0] == 0:
        return _np.zeros(0, dtype=int)
    G = _graph(adj)
    if not _nx.is_chordal(G):
        return None
    return _mcs_ordering(G)


def elimination_tree(pattern, perm) -> _np.ndarray:
    """
    The elimination tree of the Cholesky factor of the permuted pattern `A[perm][:, perm]`.

    Parameters
    ----------
    pattern : scipy.sparse matrix or numpy.ndarray
        A square matrix whose nonzero off-diagonal entries define the (symmetrized) pattern.

    perm : array_like
        The elimination ordering.

    Returns
    -------
    numpy.ndarray
        `parent[k]` is the parent of permuted row `k` (the smallest `i > k` with `L[i, k] != 0`),
        or -1 if `k` is a root. Indices refer to the permuted matrix.
    """
    adj = _adjacency(pattern)
    n = adj.shape[0]
    perm = _check_perm(perm, n)
    B = adj[perm][:, perm].tocsr()
    parent = -_np.ones(n, dtype=int)
    ancestor = -_np.ones(n, dtype=int)  # path-compressed ancestors (Liu's algorithm)
    for i in range(n):
        for k in B.indices[B.indptr[i]:B.indptr[i + 1]]:
            while k != -1 and k < i:
                next_k = ancestor[k]
                ancestor[k] = i
                if next_k == -1:
                    parent[k] = i
                k = next_k
    return parent


def symbolic_cholesky(pattern, perm) -> _sps.csc_matrix:
    """
    The sparsity pattern of the Cholesky factor of the permuted pattern `A[perm][:, perm]`.

    Parameters
    ----------
    pattern : scipy.sparse matrix or numpy.ndarray
        A square matrix whose nonzero off-diagonal entries define the (symmetrized) pattern.

    perm : array_like
        The elimination ordering.

    Returns
    -------
    scipy.sparse.csc_matrix
        A boolean lower-triangular matrix, diagonal included, whose nonzeros are those of the
        Cholesky factor `L` (fill included). Indices refer to the permuted matrix.
    """
    adj = _adjacency(pattern)
    n = adj.shape[0]
    perm = _check_perm(perm, n)
    parent = elimination_tree(adj, perm)
    B = adj[perm][:, perm].tocsr()
    rows, cols = [], []
    mark = -_np.ones(n, dtype=int)
    for i in range(n):
        # The nonzeros of row i of L are the vertices on etree paths from each k < i with B[i, k] != 0
        # up toward i: the "row subtree" of i.
        mark[i] = i
        for k in B.indices[B.indptr[i]:B.indptr[i + 1]]:
            while k < i and mark[k] != i:
                rows.append(i)
                cols.append(k)
                mark[k] = i
                k = parent[k]
    rows.extend(range(n))
    cols.extend(range(n))
    return _sps.csc_matrix((_np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))


class CholeskyStructure(object):
    """
    Reusable symbolic structure for Cholesky factorization of a symmetric sparsity pattern.

    Parameters
    ----------
    pattern : scipy.sparse matrix or numpy.ndarray
        A square matrix whose nonzero off-diagonal entries define the pattern. Numerical values
        and diagonal entries are ignored, and an edge present in either triangle is included in
        both triangles. The input is copied.

    ordering : array_like, optional
        A permutation of ``range(n)`` specifying the elimination order: ``perm[k]`` is the
        original index eliminated at step ``k``, so the factorization is of
        ``A[perm][:, perm]``. A supplied ordering is used even when it introduces avoidable fill.
        If omitted, use :func:`fill_reducing_ordering`, replacing its result with a perfect
        elimination ordering when the graph is chordal and the first ordering introduces fill.

    Attributes
    ----------
    n : int
        The number of vertices (matrix dimension).

    pattern : scipy.sparse.csr_matrix
        Symmetric Boolean off-diagonal pattern in the original vertex order.

    factor_pattern : scipy.sparse.csc_matrix
        Boolean lower-triangular Cholesky factor pattern, including its diagonal and any fill.
        Its indices refer to the permuted matrix.

    perm, inv : numpy.ndarray
        The elimination ordering and its inverse, with ``perm[inv] == arange(n)``.

    rows, cols : numpy.ndarray
        Strict-lower factor coordinates in the permuted matrix, in CSC order: columns increase
        first, and rows increase within each column. Their order also fixes the order in which
        a consumer can assign one value per factor entry.

    column_counts : numpy.ndarray
        Number of strict-lower factor entries in each column, in elimination order.

    filled_degrees : numpy.ndarray
        Vertex degrees in the undirected filled graph, in the original vertex order. The
        diagonal does not contribute to these degrees.

    Notes
    -----
    A graph is chordal exactly when it admits a perfect elimination ordering, for which there
    is no fill. Chordality alone does not make every ordering fill-free. For a nonchordal graph,
    every ordering adds fill; the factor pattern records those added entries without changing
    the requested ``pattern``. The factor pattern describes structural nonzeros: special
    numerical values can cancel entries in a particular factorization.

    The NumPy arrays, including the data, index, and pointer arrays of the two sparse matrices,
    are copied and made read-only so that ordinary array mutation cannot invalidate the cached
    relationships. Empty (zero-by-zero) patterns are supported.
    """

    def __init__(self, pattern, *, ordering=None):
        if _np.ndim(pattern) != 2:
            raise ValueError("A sparsity pattern must be a square matrix.")
        adj = _adjacency(pattern).copy()
        n = adj.shape[0]
        if ordering is None:
            perm = fill_reducing_ordering(adj)
            factor_pattern = symbolic_cholesky(adj, perm)
            if _sps.tril(factor_pattern, k=-1).nnz > _sps.tril(adj[perm][:, perm], k=-1).nnz:
                peo = perfect_elimination_ordering(adj)
                if peo is not None:
                    perm = peo
                    factor_pattern = symbolic_cholesky(adj, perm)
        else:
            perm = _np.asarray(ordering)
            if (perm.shape != (n,) or perm.dtype.kind not in 'iuf'
                    or not _np.array_equal(_np.sort(perm), _np.arange(n))):
                raise ValueError("`ordering` must be a permutation of range(%d)." % n)
            perm = _np.array(perm, dtype=int, copy=True)
            factor_pattern = symbolic_cholesky(adj, perm)

        factor_pattern = factor_pattern.copy()
        factor_pattern.sort_indices()
        lower = _sps.tril(factor_pattern, k=-1).tocsc()
        lower.sort_indices()
        self.n = n
        self.pattern = adj
        self.factor_pattern = factor_pattern
        self.perm = _np.array(perm, dtype=int, copy=True)
        self.inv = _np.argsort(self.perm)
        self.column_counts = _np.diff(lower.indptr)
        self.cols = _np.repeat(_np.arange(n), self.column_counts)
        self.rows = lower.indices.copy()
        filled_degrees = _np.bincount(self.rows, minlength=n) + self.column_counts
        self.filled_degrees = filled_degrees[self.inv]
        for array in (self.perm, self.inv, self.rows, self.cols, self.column_counts, self.filled_degrees,
                      self.pattern.data, self.pattern.indices, self.pattern.indptr,
                      self.factor_pattern.data, self.factor_pattern.indices, self.factor_pattern.indptr):
            array.flags.writeable = False
