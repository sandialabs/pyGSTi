import importlib.util

import networkx as nx
import numpy as np
import scipy.sparse as sps

from pygsti.tools.graphs import sparsechol
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
        weighted = pattern.multiply(np.arange(36).reshape(6, 6) + 1.0) + sps.dia_array((np.full((1, 6), 7), [0]),
                                                                                shape=(6, 6))
        perm = np.arange(6)
        self.assertEqual((sparsechol.symbolic_cholesky(pattern, perm)
                          != sparsechol.symbolic_cholesky(weighted, perm)).nnz, 0)
        upper_only = sps.triu(pattern)  # the pattern is symmetrized
        self.assertEqual((sparsechol.symbolic_cholesky(pattern, perm)
                          != sparsechol.symbolic_cholesky(upper_only, perm)).nnz, 0)

    def test_trivial_patterns(self):
        for n in (0, 1, 4):
            empty = sps.csr_array((n, n))
            perm = sparsechol.fill_reducing_ordering(empty)
            np.testing.assert_array_equal(np.sort(perm), np.arange(n))
            self.assertEqual(sparsechol.symbolic_cholesky(empty, perm).nnz, n)
            np.testing.assert_array_equal(sparsechol.elimination_tree(empty, perm), -np.ones(n))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            sparsechol.symbolic_cholesky(sps.csr_array((3, 4)), [0, 1, 2])
        with self.assertRaises(ValueError):
            sparsechol.elimination_tree(sps.csr_array((3, 3)), [0, 0, 1])

    def test_sparse_array_formats_preserve_symbolic_fill(self):
        # A center-first ordering fills the missing edge of this three-vertex path.
        values = np.array([[9, 2, 0], [0, 7, 3], [0, 0, 11]])
        for array_type in (sps.csr_array, sps.csc_array, sps.coo_array):
            with self.subTest(format=array_type.__name__):
                pattern = array_type(values)
                factor = sparsechol.symbolic_cholesky(pattern, [1, 0, 2])
                self.assertIsInstance(factor, sps.csc_array)
                np.testing.assert_array_equal(factor.toarray(), np.tril(np.ones((3, 3), dtype=bool)))
                np.testing.assert_array_equal(sparsechol.elimination_tree(pattern, [1, 0, 2]), [1, 2, -1])


class CholeskyStructureTester(BaseCase):

    def test_structure_matches_numeric_factor_in_supplied_order(self):
        rng = np.random.default_rng(42)
        graphs = [nx.cycle_graph(5), nx.star_graph(4), nx.path_graph(6), nx.empty_graph(4)]
        for graph in graphs:
            pattern = _pattern(graph)
            perm = rng.permutation(graph.number_of_nodes())
            structure = sparsechol.CholeskyStructure(pattern, ordering=perm)
            numeric = np.abs(_numeric_factor(pattern, perm, rng)) > 1e-12
            lower = np.tril(numeric, k=-1)
            expected_cols, expected_rows = np.nonzero(lower.T)
            filled_graph = lower | lower.T
            expected_degrees = np.empty(len(perm), dtype=int)
            expected_degrees[perm] = np.count_nonzero(filled_graph, axis=0)

            self.assertEqual(structure.n, len(perm))
            self.assertIsInstance(structure.pattern, sps.csr_array)
            self.assertIsInstance(structure.factor_pattern, sps.csc_array)
            self.assertEqual(structure.pattern.dtype, np.dtype(bool))
            self.assertEqual(structure.factor_pattern.dtype, np.dtype(bool))
            np.testing.assert_array_equal(structure.pattern.toarray(), pattern.toarray())
            np.testing.assert_array_equal(structure.factor_pattern.toarray(), numeric)
            np.testing.assert_array_equal(structure.perm, perm)
            np.testing.assert_array_equal(structure.perm[structure.inv], np.arange(len(perm)))
            np.testing.assert_array_equal(structure.rows, expected_rows)
            np.testing.assert_array_equal(structure.cols, expected_cols)
            np.testing.assert_array_equal(structure.column_counts, np.count_nonzero(lower, axis=0))
            np.testing.assert_array_equal(structure.filled_degrees, expected_degrees)

    def test_explicit_ordering_retains_fill_on_chordal_graph(self):
        # Eliminating a star's center first connects all its leaves, despite chordality.
        pattern = _pattern(nx.star_graph(3))
        structure = sparsechol.CholeskyStructure(pattern, ordering=[0, 2, 3, 1])
        np.testing.assert_array_equal(structure.perm, [0, 2, 3, 1])
        np.testing.assert_array_equal(structure.factor_pattern.toarray(), np.tril(np.ones((4, 4), bool)))
        np.testing.assert_array_equal(structure.rows, [1, 2, 3, 2, 3, 3])
        np.testing.assert_array_equal(structure.cols, [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(structure.column_counts, [3, 2, 1, 0])
        np.testing.assert_array_equal(structure.filled_degrees, [3, 3, 3, 3])

    def test_default_ordering_removes_avoidable_chordal_fill(self):
        from unittest import mock
        pattern = _pattern(nx.star_graph(4))
        # An optional ordering backend can return this valid, but poor, ordering.
        with mock.patch.object(sparsechol, 'fill_reducing_ordering', return_value=np.arange(5)):
            structure = sparsechol.CholeskyStructure(pattern)
        numeric = np.abs(_numeric_factor(pattern, structure.perm, np.random.default_rng(1))) > 1e-12
        self.assertEqual(np.count_nonzero(np.tril(numeric, k=-1)), 4)
        np.testing.assert_array_equal(structure.factor_pattern.toarray(), numeric)
        np.testing.assert_array_equal(structure.filled_degrees, [4, 1, 1, 1, 1])

    def test_nonchordal_default_keeps_backend_order_and_required_fill(self):
        from unittest import mock
        pattern = _pattern(nx.cycle_graph(4))
        perm = np.array([2, 0, 3, 1])
        with mock.patch.object(sparsechol, 'fill_reducing_ordering', return_value=perm):
            structure = sparsechol.CholeskyStructure(pattern)
        np.testing.assert_array_equal(structure.perm, perm)
        np.testing.assert_array_equal(structure.rows, [2, 3, 2, 3, 3])
        np.testing.assert_array_equal(structure.cols, [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(structure.column_counts, [2, 2, 1, 0])
        np.testing.assert_array_equal(structure.filled_degrees, [2, 3, 2, 3])

    def test_pattern_normalization_ignores_values_and_diagonal(self):
        pattern = np.array([[9.0, 2.0, 0.0], [0.0, -7.0, -3.0], [0.0, 0.0, 5.0]])
        structure = sparsechol.CholeskyStructure(pattern, ordering=[0, 1, 2])
        np.testing.assert_array_equal(structure.pattern.toarray(),
                                      [[False, True, False], [True, False, True], [False, True, False]])
        np.testing.assert_array_equal(structure.factor_pattern.toarray(),
                                      [[True, False, False], [True, True, False], [False, True, True]])

    def test_structural_arrays_are_copied_and_read_only(self):
        pattern = sps.csr_array([[False, True, False], [True, False, True], [False, True, False]])
        perm = np.array([1, 2, 0])
        expected_pattern = pattern.toarray()
        structure = sparsechol.CholeskyStructure(pattern, ordering=perm)
        pattern.data[:] = False
        perm[:] = [0, 1, 2]
        np.testing.assert_array_equal(structure.pattern.toarray(), expected_pattern)
        np.testing.assert_array_equal(structure.perm, [1, 2, 0])

        arrays = [structure.perm, structure.inv, structure.rows, structure.cols,
                  structure.column_counts, structure.filled_degrees]
        for matrix in (structure.pattern, structure.factor_pattern):
            arrays.extend([matrix.data, matrix.indices, matrix.indptr])
        for array in arrays:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0

    def test_empty_structure(self):
        for ordering in (None, []):
            structure = sparsechol.CholeskyStructure(sps.csr_array((0, 0)), ordering=ordering)
            self.assertEqual(structure.n, 0)
            for matrix in (structure.pattern, structure.factor_pattern):
                self.assertEqual(matrix.shape, (0, 0))
                self.assertEqual(matrix.nnz, 0)
            for array in (structure.perm, structure.inv, structure.rows, structure.cols,
                          structure.column_counts, structure.filled_degrees):
                self.assertEqual(array.shape, (0,))
                self.assertFalse(array.flags.writeable)

    def test_rejects_nonmatrix_and_nonsquare_patterns(self):
        patterns = [np.ones((3, 4)), sps.csr_array((3, 4)), np.ones(1), np.ones((1, 1, 1)), 1.0]
        for pattern in patterns:
            with self.subTest(shape=np.shape(pattern)):
                with self.assertRaises(ValueError):
                    sparsechol.CholeskyStructure(pattern)

    def test_rejects_invalid_orderings(self):
        orderings = [[0, 0, 1], [0, 1], [0, 1, 3], [-1, 0, 1], [[0, 1, 2]],
                     [0.5, 1, 2], ['0', '1', '2'], [False, True, True]]
        for ordering in orderings:
            with self.subTest(ordering=ordering):
                with self.assertRaises(ValueError):
                    sparsechol.CholeskyStructure(np.zeros((3, 3)), ordering=ordering)


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
