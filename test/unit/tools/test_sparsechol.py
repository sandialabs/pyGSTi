import importlib.util

import networkx as nx
import numpy as np
import scipy.sparse as sps

from pygsti.tools import sparsechol
from ..util import BaseCase


def _pattern(G):
    """ A symmetric boolean adjacency matrix for the networkx graph G on nodes 0..n-1. """
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format='csr') != 0


def _numeric_factor(pattern, perm, rng):
    """ The numerical Cholesky factor of a random positive definite matrix with the given pattern. """
    n = pattern.shape[0]
    A = sps.triu(pattern, k=1).toarray() * rng.uniform(0.5, 1.5, size=(n, n))
    A = A + A.T + np.diag(np.full(n, n + 1.0))
    return np.linalg.cholesky(A[np.ix_(perm, perm)])


def _fill(pattern, perm):
    L = sparsechol.symbolic_cholesky(pattern, perm)
    B = pattern[perm][:, perm]
    return sps.tril(L, k=-1).nnz - sps.tril(B, k=-1).nnz


def _bridged_cliques():
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    G.add_edges_from([(10, 0), (10, 5)])
    return G


class SymbolicCholeskyTester(BaseCase):

    def test_symbolic_pattern_and_etree_match_numeric_factor(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            n = int(rng.integers(1, 16))
            G = nx.gnp_random_graph(n, rng.uniform(0.1, 0.5), seed=int(rng.integers(1 << 30)))
            pattern = _pattern(G)
            perm = rng.permutation(n)
            L = _numeric_factor(pattern, perm, rng)
            numeric = np.abs(L) > 1e-12
            symbolic = sparsechol.symbolic_cholesky(pattern, perm).toarray()
            np.testing.assert_array_equal(symbolic, numeric)

            parent = sparsechol.elimination_tree(pattern, perm)
            expected = [min([i for i in range(k + 1, n) if numeric[i, k]], default=-1) for k in range(n)]
            np.testing.assert_array_equal(parent, expected)

    def test_values_and_diagonal_are_ignored(self):
        G = nx.cycle_graph(6)
        pattern = _pattern(G).astype(float)
        weighted = pattern.multiply(np.arange(36).reshape(6, 6) + 1.0) + sps.identity(6) * 7
        perm = np.arange(6)
        self.assertEqual((sparsechol.symbolic_cholesky(pattern, perm)
                          != sparsechol.symbolic_cholesky(weighted, perm)).nnz, 0)
        upper_only = sps.triu(pattern)  # the pattern is symmetrized
        self.assertEqual((sparsechol.symbolic_cholesky(pattern, perm)
                          != sparsechol.symbolic_cholesky(upper_only, perm)).nnz, 0)

    def test_trivial_patterns(self):
        for n in (0, 1, 4):
            empty = sps.csr_matrix((n, n))
            perm = sparsechol.fill_reducing_ordering(empty)
            np.testing.assert_array_equal(np.sort(perm), np.arange(n))
            self.assertEqual(sparsechol.symbolic_cholesky(empty, perm).nnz, n)
            np.testing.assert_array_equal(sparsechol.elimination_tree(empty, perm), -np.ones(n))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            sparsechol.symbolic_cholesky(sps.csr_matrix((3, 4)), [0, 1, 2])
        with self.assertRaises(ValueError):
            sparsechol.elimination_tree(sps.csr_matrix((3, 3)), [0, 0, 1])


class OrderingTester(BaseCase):

    BACKEND_MODULES = {'cholmod': 'sksparse', 'qdldl': 'qdldl', 'networkx': 'networkx'}

    def _available_backends(self):
        return [b for b, mod in self.BACKEND_MODULES.items() if importlib.util.find_spec(mod) is not None]

    def test_backends_return_permutations(self):
        G = nx.gnp_random_graph(30, 0.2, seed=1)
        for backend in self._available_backends():
            perm = sparsechol.fill_reducing_ordering(_pattern(G), _backend=backend)
            np.testing.assert_array_equal(np.sort(perm), np.arange(30))

    def test_amd_backends_eliminate_arrow_hub_last(self):
        pattern = _pattern(nx.star_graph(7))  # hub 0 joined to 1..7
        for backend in [b for b in self._available_backends() if b != 'networkx']:
            perm = sparsechol.fill_reducing_ordering(pattern, _backend=backend)
            self.assertEqual(perm[-1], 0)
            self.assertEqual(_fill(pattern, perm), 0)

    def test_networkx_fallback_has_no_fill_on_chordal_patterns(self):
        rng = np.random.default_rng(2)
        chordal = [_bridged_cliques(), nx.star_graph(7), nx.complete_graph(6), nx.path_graph(9)]
        for _ in range(5):  # random interval graphs are chordal
            ivals = np.sort(rng.uniform(size=(12, 2)), axis=1)
            G = nx.Graph()
            G.add_nodes_from(range(12))
            G.add_edges_from((i, j) for i in range(12) for j in range(i + 1, 12)
                             if ivals[i, 0] <= ivals[j, 1] and ivals[j, 0] <= ivals[i, 1])
            chordal.append(G)
        for G in chordal:
            self.assertTrue(nx.is_chordal(G))
            perm = sparsechol.fill_reducing_ordering(_pattern(G), _backend='networkx')
            self.assertEqual(_fill(_pattern(G), perm), 0)

    def test_perfect_elimination_ordering(self):
        G = _bridged_cliques()
        perm = sparsechol.perfect_elimination_ordering(_pattern(G))
        self.assertEqual(_fill(_pattern(G), perm), 0)
        self.assertIsNone(sparsechol.perfect_elimination_ordering(_pattern(nx.cycle_graph(4))))

    def test_default_ordering_is_valid(self):
        G = nx.gnp_random_graph(25, 0.15, seed=3)
        perm = sparsechol.fill_reducing_ordering(_pattern(G))
        np.testing.assert_array_equal(np.sort(perm), np.arange(25))

    def test_networkx_fallback_fill_is_a_minimal_triangulation(self):
        # Every minimal triangulation of an n-cycle adds exactly n - 3 chords.
        for n in (4, 5, 8, 13):
            pattern = _pattern(nx.cycle_graph(n))
            perm = sparsechol.fill_reducing_ordering(pattern, _backend='networkx')
            self.assertEqual(_fill(pattern, perm), n - 3)

    def test_installed_but_broken_backend_warns_and_falls_back(self):
        # A missing module is skipped silently; an installed but incompatible one warns, then is skipped.
        from unittest import mock
        spec = importlib.util.find_spec('networkx')  # any non-None spec
        def broken(adj):
            raise ImportError("cannot import name 'analyze'")
        pattern = _pattern(nx.star_graph(4))
        with mock.patch.object(sparsechol._importlib_util, 'find_spec', return_value=spec), \
             mock.patch.object(sparsechol, '_ordering_cholmod', broken), \
             mock.patch.object(sparsechol, '_ordering_qdldl', broken):
            with self.assertWarns(RuntimeWarning):
                perm = sparsechol.fill_reducing_ordering(pattern)
        np.testing.assert_array_equal(np.sort(perm), np.arange(5))
        self.assertEqual(_fill(pattern, perm), 0)  # the networkx fallback
