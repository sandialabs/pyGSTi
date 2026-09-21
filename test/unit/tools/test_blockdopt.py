"""Tests for the greedy block D-optimal kernel in pygsti.tools.edesigntools.blockdopt.

The kernel picks candidates by comparing R-factor diagonals from a Householder
QR.  Everything here judges its choices with `greedy_candidate_scores` and
`greedy_path_log_volumes` instead, which get the same quantity from a float64
Cholesky factorization of the information matrix and share no arithmetic with
the kernel.
A test that scored the kernel with the kernel's own numbers would pass on a
consistently wrong implementation.

The correctness argument rests on two standard references, both of them plain
numpy and scipy.  `ColumnPivotedQRTester` uses the fact that at
`block_size == 1` this algorithm *is* Businger-Golub column-pivoted QR of an
augmented matrix, so `scipy.linalg.qr(..., pivoting=True)` gives the exact
answer to compare against.  `BruteForceEnumerationTester` enumerates small
problems outright, evaluating the objective with `slogdet` and reading the
greedy ordering off the resulting table.

Ported from `python-block-edesigns/tests/test_reference_impls.py`, minus the
cases that compare against that package's compiled C++ kernel.

From `JacobianFlatteningTester` down, the tests cover what sits between a pyGSTi
model and the kernel: flattening a `bulk_dprobs` result into the kernel's
matrix, perturbing a target model so that its Jacobian is representative, the
function-shaped entry points `rank_circuits_by_dopt` and
`reduce_design_by_dopt`, and `BlockDoptReducer`, the `DesignReducer` that wraps
them.  The `DesignReducer` contract itself is tested in test_reduction.py.
"""
import itertools
import unittest.mock
import warnings

import numpy as np
import scipy.linalg

from pygsti.tools.edesigntools import blockdopt as bd

from ..util import BaseCase


def _design(seed, ncand, b, p, dtype=np.float64, order="C"):
    """Random `A = J^T` of shape `(p, ncand * b)` in the requested layout."""
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((ncand * b, p))
    return np.array(J.T, dtype=dtype, order=order)


class BlockDoptKernelTester(BaseCase):

    def assert_tol_greedy_path(self, A, b, piv, rel_tol=1e-8, label="selection"):
        """Every pick must be within tolerance of the best score given its own prefix.

        This is the definition of greedy optimality, checked step by step
        against the independent scorer.
        """
        for k in range(len(piv)):
            s = bd.greedy_candidate_scores(A, b, piv[:k])
            best = np.max(s)
            tol = rel_tol * max(1.0, abs(best))
            self.assertGreaterEqual(
                s[piv[k]], best - tol,
                f"{label}: pick {piv[k]} at step {k} scores {s[piv[k]]!r}, but "
                f"{int(np.argmax(s))} scores {best!r} (tol {tol:g})")

    # -- the objective ----------------------------------------------------- #

    def test_scores_are_the_cumulative_log_volume_curve(self):
        """The per-step winning scores equal 0.5*logdet(I + J_S^T J_S)."""
        A = _design(30, 24, 4, 12)
        piv, scores = bd.block_linear_dopt(A, 4, 8)
        curve = bd.greedy_path_log_volumes(A, 4, piv)
        self.assertArraysAlmostEqual(curve[1:], scores, places=8)
        self.assertEqual(curve[0], 0.0)
        # Adding blocks can only add information.
        self.assertTrue(np.all(np.diff(curve) >= -1e-12))

    def test_scores_do_not_drift_over_a_full_ranking(self):
        """A long run is the test of the in-place `G <- G C^-1` updates.

        The kernel carries the transformed candidate matrix and overwrites it
        once per step, so a full ranking applies as many updates as there are
        candidates.  Each `C^-1` is a contraction, so the error should not
        accumulate -- checked against the independent log-volume curve, which
        recomputes from the untouched input every time.
        """
        A = _design(40, 120, 4, 16)
        piv, scores = bd.block_linear_dopt(A, 4, 120)
        self.assertEqual(len(piv), 120)
        self.assertEqual(len(set(piv.tolist())), 120)        # each candidate once
        curve = bd.greedy_path_log_volumes(A, 4, piv)
        self.assertArraysAlmostEqual(curve[1:], scores, places=8)

    def test_selection_stays_greedy_optimal_over_a_full_ranking(self):
        """Greedy optimality at every one of 120 steps, not just the first few.

        Retiring a winner swaps it past the active boundary, so after the first
        step the surviving candidates are no longer in ascending order.  This
        walks the whole ranking to catch a bookkeeping error that a short run
        would miss.
        """
        A = _design(41, 120, 4, 16)
        piv, _ = bd.block_linear_dopt(A, 4, 120)
        self.assert_tol_greedy_path(A, 4, piv, label="full ranking")

    def test_duplicate_blocks_tie_break_by_original_index_late_in_the_ranking(self):
        """Ties must resolve to the lowest *original* index after compaction.

        Companion to the short duplicate test: here the duplicated pairs are
        weak blocks that are only reached once many swaps have permuted the
        active set, so a tie-break that used the current position rather than
        the original index would pick the wrong one.
        """
        b, ncand, p = 4, 60, 12
        A = _design(42, ncand, b, p)
        A[:, 40 * b:] *= 1e-3                        # make the tail get picked last
        dups = [(41, 55), (44, 58), (47, 52)]
        for lo, hi in dups:
            A[:, hi * b:(hi + 1) * b] = A[:, lo * b:(lo + 1) * b]
        piv, _ = bd.block_linear_dopt(A, b, ncand)
        order = {int(blk): pos for pos, blk in enumerate(piv)}
        for lo, hi in dups:
            self.assertGreater(order[lo], 20, (lo, order[lo]))   # genuinely late
            self.assertLess(order[lo], order[hi], (lo, hi, piv))

    def test_first_pick_is_the_standalone_argmax(self):
        A = _design(31, 15, 6, 20)
        piv, _ = bd.block_linear_dopt(A, 6, 1)
        self.assertEqual(piv.tolist(), [int(np.argmax(bd.greedy_candidate_scores(A, 6)))])

    def test_selection_is_greedy_optimal_at_every_step(self):
        for seed, ncand, b, p, k in [(0, 20, 4, 12, 8),
                                     (1, 30, 6, 16, 10),
                                     (2, 25, 1, 10, 12),      # scalar blocks
                                     (3, 12, 8, 3, 6)]:       # block_size > num params
            with self.subTest(seed=seed):
                A = _design(seed, ncand, b, p)
                piv, _ = bd.block_linear_dopt(A, b, k)
                self.assertEqual(len(piv), min(ncand, k))
                self.assert_tol_greedy_path(A, b, piv)

    def test_ill_conditioned_input_stays_greedy_optimal(self):
        """Polynomially decaying singular values make later gains tiny (near-ties)."""
        rng = np.random.default_rng(12)
        ncand, b, p = 24, 4, 12
        U, _, Vt = np.linalg.svd(rng.standard_normal((ncand * b, p)), full_matrices=False)
        J = (U * (1 + np.arange(p)) ** -1.5) @ Vt
        A = np.ascontiguousarray(J.T)
        piv, _ = bd.block_linear_dopt(A, b, ncand)
        self.assert_tol_greedy_path(A, b, piv, rel_tol=1e-6)

    def test_float32_selection_is_greedy_optimal_to_float32_tolerance(self):
        A = _design(1, 30, 6, 16, dtype=np.float32)
        piv, _ = bd.block_linear_dopt(A, 6, 10)
        self.assert_tol_greedy_path(A, 6, piv, rel_tol=1e-3)

    # -- determinism and purity -------------------------------------------- #

    def test_is_deterministic_and_does_not_modify_input(self):
        A = _design(32, 12, 3, 9)
        A0 = A.copy()
        p1, _ = bd.block_linear_dopt(A, 3, 5)
        p2, _ = bd.block_linear_dopt(A, 3, 5)
        self.assertArraysEqual(p1, p2)
        self.assertArraysEqual(A, A0)

    def test_noncontiguous_input_gives_the_same_answer_as_a_copy(self):
        base = _design(11, 40, 4, 12)
        A = base[:, ::2]                            # 20 candidates, non-contiguous
        self.assertFalse(A.flags.c_contiguous or A.flags.f_contiguous)
        self.assertArraysEqual(bd.block_linear_dopt(A, 4, 8)[0],
                               bd.block_linear_dopt(np.ascontiguousarray(A), 4, 8)[0])

    def test_fortran_order_input_gives_the_same_answer(self):
        c = _design(0, 20, 4, 12, order="C")
        f = np.asfortranarray(c)
        self.assertArraysEqual(bd.block_linear_dopt(c, 4, 8)[0], bd.block_linear_dopt(f, 4, 8)[0])

    def test_exact_duplicate_blocks_tie_break_to_the_lowest_index(self):
        """Bitwise-identical blocks tie exactly; argmax takes the first maximum."""
        b, ncand, p = 4, 16, 10
        A = _design(20, ncand, b, p)
        dups = [(2, 9), (5, 13)]
        for lo, hi in dups:
            A[:, hi * b:(hi + 1) * b] = A[:, lo * b:(lo + 1) * b]
        piv, _ = bd.block_linear_dopt(A, b, ncand)
        order = {int(blk): pos for pos, blk in enumerate(piv)}
        for lo, hi in dups:
            self.assertLess(order[lo], order[hi], (lo, hi, piv))

    # -- the scipy-version fallback ---------------------------------------- #

    def test_scipy_is_actually_batched_here(self):
        """Not a requirement -- a canary, so a fallback-only run is visible."""
        if not bd._SCIPY_QR_IS_BATCHED:
            self.skipTest("this SciPy does not accept a stack of matrices in linalg.qr")

    def test_loop_fallback_selects_identically_to_the_batched_path(self):
        """The scipy<1.18 path must not change any answer."""
        if not bd._SCIPY_QR_IS_BATCHED:
            self.skipTest("already running the fallback; nothing to compare against")
        A = _design(7, 22, 5, 14)
        batched = bd.block_linear_dopt(A, 5, 11)
        with unittest.mock.patch.object(bd, '_SCIPY_QR_IS_BATCHED', False):
            looped = bd.block_linear_dopt(A, 5, 11)
        self.assertArraysEqual(batched[0], looped[0])
        self.assertArraysEqual(batched[1], looped[1])

    def test_batched_qr_probe_rejects_a_nonbatched_qr(self):
        """The probe must return False, not raise, when qr ignores the stack."""
        import scipy.linalg
        with unittest.mock.patch.object(scipy.linalg, 'qr',
                                        side_effect=ValueError("expected matrix")):
            self.assertFalse(bd._scipy_qr_is_batched())

    # -- edge cases and error paths ---------------------------------------- #

    def test_max_blocks_zero_returns_empty(self):
        A = _design(40, 8, 2, 5)
        piv, scores = bd.block_linear_dopt(A, 2, 0)
        self.assertEqual(piv.shape, (0,))
        self.assertEqual(piv.dtype, np.int64)
        self.assertEqual(scores.shape, (0,))

    def test_max_blocks_is_clamped_to_the_candidate_count(self):
        A = _design(41, 5, 3, 7)
        self.assertEqual(len(bd.block_linear_dopt(A, 3, 100)[0]), 5)

    def test_invalid_arguments_raise(self):
        A = _design(42, 8, 3, 6)             # n = 24
        for kwargs in [dict(block_size=7),   # 24 % 7 != 0
                       dict(block_size=0),
                       dict(block_size=-2),
                       dict(max_blocks=-1)]:
            with self.subTest(**kwargs):
                args = dict(block_size=3, max_blocks=4)
                args.update(kwargs)
                with self.assertRaises(ValueError):
                    bd.block_linear_dopt(A, args["block_size"], args["max_blocks"])

    def test_bad_dtype_and_ndim_raise(self):
        with self.assertRaises(TypeError):
            bd.block_linear_dopt(np.arange(24, dtype=np.int64).reshape(4, 6), 2, 2)
        with self.assertRaises(ValueError):
            bd.block_linear_dopt(np.ones(6), 2, 2)

    def test_nonfinite_input_is_rejected(self):
        """NaN and Inf are caller errors, refused before any selection happens."""
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                A = _design(43, 6, 4, 10)
                A[0, 5] = bad
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    bd.block_linear_dopt(A, 4, 3)

    def test_information_free_blocks_gain_nothing_and_rank_last(self):
        """The ridged objective orders degenerate blocks; it does not reject them.

        `0.5*logdet(I_b + G_i G_i^T)` is finite and nonnegative for every
        candidate.  A zero block gains exactly zero, a rank-deficient nonzero
        block gains something positive but less than a full-rank one, and both
        therefore sort behind every block that carries information.  There is
        no singular case and nothing for the kernel to skip.
        """
        b, p = 3, 6
        A = _design(45, 6, b, p)
        A[:, 3 * b:4 * b] = 0.0                              # candidate 3: zero
        u = A[:, 0].copy()
        A[:, 4 * b:5 * b] = np.outer(u, [1.0, 0.5, -0.25])   # candidate 4: rank 1

        standalone = bd.greedy_candidate_scores(A, b)
        self.assertEqual(standalone[3], 0.0)
        self.assertGreater(standalone[4], 0.0)
        self.assertTrue(np.all(np.isfinite(standalone)))

        piv, scores = bd.block_linear_dopt(A, b, 6)
        self.assertEqual(sorted(piv.tolist()), list(range(6)))
        self.assertEqual(piv.tolist()[-1], 3)                # the zero block last
        self.assertEqual(piv.tolist()[-2], 4)                # the rank-1 block next
        gains = np.diff(np.concatenate([[0.0], scores]))
        self.assertAlmostEqual(gains[-1], 0.0, places=12)
        self.assertGreater(gains[-2], 0.0)
        self.assertTrue(np.all(gains >= -1e-12))


class GreedyScorerTester(BaseCase):

    def test_candidate_scores_mark_selected_blocks_as_minus_inf(self):
        A = _design(50, 10, 2, 6)
        s = bd.greedy_candidate_scores(A, 2, selected=(3, 7))
        self.assertEqual(s[3], -np.inf)
        self.assertEqual(s[7], -np.inf)
        self.assertTrue(np.all(np.isfinite(np.delete(s, [3, 7]))))

    def test_candidate_score_matches_an_explicit_determinant(self):
        A = _design(51, 6, 2, 4)
        s = bd.greedy_candidate_scores(A, 2, selected=(1,))
        blocks = [A[:, i * 2:(i + 1) * 2] for i in range(6)]
        expected = 0.5 * np.log(np.linalg.det(
            np.eye(4) + blocks[1] @ blocks[1].T + blocks[4] @ blocks[4].T))
        self.assertAlmostEqual(s[4], expected, places=10)

    def test_path_log_volumes_start_at_zero_and_have_the_right_length(self):
        A = _design(52, 8, 3, 5)
        out = bd.greedy_path_log_volumes(A, 3, [2, 5, 0])
        self.assertEqual(out.shape, (4,))
        self.assertEqual(out[0], 0.0)

    def test_path_log_volumes_of_an_empty_selection(self):
        A = _design(53, 8, 3, 5)
        self.assertArraysEqual(bd.greedy_path_log_volumes(A, 3, []), np.zeros(1))

    def test_path_log_volumes_are_order_independent_at_the_end(self):
        """The final log-volume depends on the set, not the order it was built in."""
        A = _design(54, 8, 3, 5)
        a = bd.greedy_path_log_volumes(A, 3, [1, 4, 6, 0])[-1]
        b = bd.greedy_path_log_volumes(A, 3, [6, 0, 1, 4])[-1]
        self.assertAlmostEqual(a, b, places=10)

    def test_greedy_selection_beats_a_random_subset_on_log_volume(self):
        """Not guaranteed in general, but overwhelmingly true and worth pinning:
        if greedy stops beating random draws, the objective is not being
        maximized."""
        A = _design(55, 40, 4, 10)
        k = 8
        piv, _ = bd.block_linear_dopt(A, 4, k)
        greedy = bd.greedy_path_log_volumes(A, 4, piv)[-1]
        rng = np.random.default_rng(0)
        for trial in range(10):
            subset = rng.choice(40, size=k, replace=False)
            with self.subTest(trial=trial):
                self.assertGreater(greedy, bd.greedy_path_log_volumes(A, 4, subset)[-1])


# --------------------------------------------------------------------------- #
#  Standard references
# --------------------------------------------------------------------------- #


def _scaled_design(seed, m, n, scaling):
    """Random `(m, n)` matrix whose columns are scaled to make ties unlikely."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    if scaling == "decay":
        A *= 10.0 ** np.linspace(0.0, -2.0, n)
    elif scaling == "spiky":
        A[:, ::3] *= 50.0
    elif scaling != "flat":
        raise ValueError(scaling)
    return np.ascontiguousarray(A)


class ColumnPivotedQRTester(BaseCase):
    """At `block_size == 1` the kernel must reproduce LAPACK's pivoted QR.

    Sylvester's determinant identity gives

        0.5*logdet(I_m + A_S A_S^T) = 0.5*logdet(I_|S| + A_S^T A_S)
                                    = 0.5*logdet(Aug_S^T Aug_S),

    where `Aug = vstack([A, eye(n)])` and `Aug_S` is its columns `S`.  So the
    ridged objective is the log-volume of a set of columns of `Aug`, and
    greedy log-volume maximization over columns is by definition the
    Businger-Golub column-pivoted QR of `Aug`.  `scipy.linalg.qr(Aug,
    pivoting=True)` is therefore an exact reference for both the selection
    order and the score curve, with no scaling trick and no tolerance beyond
    rounding.  `Aug` always has full column rank, so the reference is defined
    even where `A` is rank-deficient, which is the same thing the unit ridge
    does for the kernel.
    """

    # (seed, m, n, scaling); m < n cases make A rank-deficient partway through.
    PROBLEMS = [
        (200, 2, 3, "flat"),
        (201, 3, 8, "flat"),
        (202, 3, 8, "decay"),
        (203, 5, 14, "flat"),
        (204, 5, 14, "spiky"),
        (205, 8, 10, "decay"),
        (206, 4, 4, "flat"),
        (207, 6, 7, "spiky"),
    ]

    @staticmethod
    def _reference(A):
        """Pivots and cumulative log-volume curve from `scipy.linalg.qr`."""
        aug = np.vstack([A, np.eye(A.shape[1], dtype=A.dtype)])
        _, R, pivots = scipy.linalg.qr(aug, pivoting=True, mode='economic')
        return pivots, np.cumsum(np.log(np.abs(np.diag(R))))

    def _check_every_budget(self, A, places=9):
        pivots, curve = self._reference(A)
        for k in range(1, A.shape[1] + 1):
            piv, scores = bd.block_linear_dopt(A, 1, k)
            self.assertArraysEqual(piv, pivots[:k])
            self.assertArraysAlmostEqual(scores, curve[:k], places=places)

    def test_the_kernel_is_pivoted_qr_on_the_augmented_matrix(self):
        for seed, m, n, scaling in self.PROBLEMS:
            with self.subTest(seed=seed, scaling=scaling):
                self._check_every_budget(_scaled_design(seed, m, n, scaling))

    def test_it_is_still_pivoted_qr_on_the_loop_fallback(self):
        """Same comparison with the batched-QR path switched off."""
        if not bd._SCIPY_QR_IS_BATCHED:
            self.skipTest("already running the fallback; the test above covered it")
        with unittest.mock.patch.object(bd, '_SCIPY_QR_IS_BATCHED', False):
            for seed, m, n, scaling in self.PROBLEMS:
                with self.subTest(seed=seed, scaling=scaling):
                    self._check_every_budget(_scaled_design(seed, m, n, scaling))

    def test_float32_matches_the_reference_on_a_well_separated_problem(self):
        """Exact pivots where the gaps are wide; the curve to float32 precision."""
        A = _scaled_design(208, 5, 9, "decay").astype(np.float32)
        self._check_every_budget(A, places=4)


class BruteForceEnumerationTester(BaseCase):
    """For small problems, the greedy ordering is computable by enumeration.

    `_subset_log_volumes` evaluates `0.5*logdet(I_m + sum_{i in S} A_i A_i^T)`
    directly with `numpy.linalg.slogdet` for every subset `S`, walking every
    ordering of the candidates so that the objective's independence of the
    order blocks are added in is checked rather than assumed.
    `_brute_force_greedy` then reads the greedy ordering straight off that
    table.  Nothing here touches the kernel's arithmetic, its QR, or its
    determinant-lemma reduction.
    """

    @staticmethod
    def _subset_log_volumes(A, block_size, test_case=None):
        """`frozenset(indices) -> 0.5*logdet(I_m + sum A_i A_i^T)`, by enumeration."""
        m, n = A.shape
        n_cand = n // block_size
        blocks = [A[:, i * block_size:(i + 1) * block_size] for i in range(n_cand)]

        def value(subset):
            M = np.eye(m)
            for i in sorted(subset):                    # canonical summation order
                M = M + blocks[i] @ blocks[i].T
            return 0.5 * np.linalg.slogdet(M)[1]

        table = {frozenset(): 0.0}
        for order in itertools.permutations(range(n_cand)):
            running = np.eye(m)
            for depth in range(1, n_cand + 1):
                subset = frozenset(order[:depth])
                running = running + blocks[order[depth - 1]] @ blocks[order[depth - 1]].T
                if subset not in table:
                    table[subset] = value(subset)
                # This ordering's own running sum must agree with the canonical one.
                if test_case is not None:
                    test_case.assertAlmostEqual(
                        0.5 * np.linalg.slogdet(running)[1], table[subset], places=10)
        return table

    @staticmethod
    def _brute_force_greedy(table, n_cand, max_blocks):
        """Greedy over the enumerated table: argmax gain, lowest index on exact ties."""
        selected, curve = [], []
        current = table[frozenset()]
        for _ in range(min(n_cand, max_blocks)):
            best_gain, best_i, best_value = -np.inf, None, None
            for i in range(n_cand):                     # ascending, so ties go low
                if i in selected:
                    continue
                candidate_value = table[frozenset(selected + [i])]
                if candidate_value - current > best_gain:
                    best_gain, best_i, best_value = candidate_value - current, i, candidate_value
            selected.append(best_i)
            current = best_value
            curve.append(current)
        return selected, np.array(curve)

    def _compare(self, A, block_size, max_blocks=None):
        n_cand = A.shape[1] // block_size
        max_blocks = n_cand if max_blocks is None else max_blocks
        table = self._subset_log_volumes(A, block_size, test_case=self)
        want_piv, want_curve = self._brute_force_greedy(table, n_cand, max_blocks)
        piv, scores = bd.block_linear_dopt(A, block_size, max_blocks)
        self.assertEqual(piv.tolist(), want_piv)
        self.assertArraysAlmostEqual(scores, want_curve, places=10)

    def test_the_kernel_matches_brute_force(self):
        # (seed, n_candidates, block_size, m)
        for seed, n_cand, b, m in [(300, 4, 1, 3),
                                   (301, 6, 1, 2),      # more candidates than params
                                   (302, 5, 2, 4),
                                   (303, 3, 2, 6),
                                   (304, 6, 2, 3),
                                   (305, 4, 3, 5),
                                   (306, 5, 3, 6)]:
            with self.subTest(seed=seed, block_size=b):
                self._compare(_scaled_design(seed, m, n_cand * b, "flat"), b)

    def test_it_matches_brute_force_on_a_partial_budget(self):
        self._compare(_scaled_design(307, 5, 12, "spiky"), 2, max_blocks=3)

    def _assert_greedy_up_to_ties(self, A, block_size, tol=1e-9):
        """The kernel's ordering is *a* greedy ordering: at every step its pick is
        within `tol` of the best available gain, and its score is that prefix's
        log-volume.  Which of several tied maximizers it takes is not checked here,
        because the enumerated table itself is subject to rounding and cannot
        adjudicate an exact tie; `BlockDoptKernelTester` pins the tie-break rule."""
        n_cand = A.shape[1] // block_size
        table = self._subset_log_volumes(A, block_size, test_case=self)
        piv, scores = bd.block_linear_dopt(A, block_size, n_cand)
        self.assertEqual(sorted(piv.tolist()), list(range(n_cand)))
        selected = []
        for step, i in enumerate(piv.tolist()):
            best = max(table[frozenset(selected + [j])] for j in range(n_cand) if j not in selected)
            chosen = table[frozenset(selected + [i])]
            self.assertGreaterEqual(chosen, best - tol)
            self.assertAlmostEqual(scores[step], chosen, places=10)
            selected.append(i)

    def test_duplicated_blocks_give_one_of_the_tied_greedy_orderings(self):
        """Candidates 1 and 3 are bitwise equal, so two greedy orderings are valid.

        Either is accepted: the brute-force table sums the two tied subsets in
        different orders, so a strict comparison there could break the tie by
        rounding and fail spuriously.
        """
        b = 2
        A = _scaled_design(308, 4, 5 * b, "flat")
        A[:, 3 * b:4 * b] = A[:, 1 * b:2 * b]           # candidate 3 duplicates 1
        self._assert_greedy_up_to_ties(A, b)
        piv = bd.block_linear_dopt(A, b, 5)[0].tolist()
        self.assertLess(piv.index(1), piv.index(3))     # the kernel's own rule, exact ties

    def test_it_matches_brute_force_with_degenerate_blocks(self):
        """A zero block and a rank-1 block, ranked by the same objective."""
        b = 3
        A = _scaled_design(309, 5, 5 * b, "flat")
        A[:, 2 * b:3 * b] = 0.0                          # candidate 2: zero
        A[:, 4 * b:5 * b] = np.outer(A[:, 0], [1.0, -2.0, 0.5])   # candidate 4: rank 1
        self._compare(A, b)


class JacobianFlatteningTester(BaseCase):
    """`jacobian_dict_to_array` turns a bulk_dprobs result into the kernel's `A`."""

    @staticmethod
    def _jac_dict(num_circuits, num_outcomes, num_params, start=0.0):
        v = start
        out = {}
        for c in range(num_circuits):
            per_circuit = {}
            for o in range(num_outcomes):
                per_circuit['o%d' % o] = np.full(num_params, v)
                v += 1.0
            out['c%d' % c] = per_circuit
        return out

    def test_shape_and_block_size(self):
        jac, block_size = bd.jacobian_dict_to_array(self._jac_dict(5, 4, 7))
        self.assertEqual(jac.shape, (20, 7))
        self.assertEqual(block_size, 4)

    def test_rows_are_grouped_per_circuit_in_key_order(self):
        # The block order follows the dict's keys, not any input circuit list:
        # bulk_dprobs may deduplicate and reorder, so this is the mapping a
        # caller has to use to get back from a block index to a circuit.
        jac, block_size = bd.jacobian_dict_to_array(self._jac_dict(3, 2, 1))
        self.assertArraysEqual(jac.ravel(), np.arange(6.0))

    def test_ragged_outcome_counts_raise(self):
        ragged = self._jac_dict(2, 3, 4)
        del ragged['c1']['o2']
        with self.assertRaises(ValueError) as ctx:
            bd.jacobian_dict_to_array(ragged)
        self.assertIn('uniform outcome count', str(ctx.exception))

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            bd.jacobian_dict_to_array({})


class _ModelFixture:
    """A 1-qubit H+S model and a handful of circuits to differentiate on.

    Class-scoped: nothing below mutates the model, and `perturb_errorgen_rates`
    is required to copy.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from pygsti.modelpacks import smq1Q_XYI
        cls.target = smq1Q_XYI.target_model('H+S')
        edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=2)
        cls.circuits = list(edesign.all_circuits_needing_data)[:40]
        # pyGSTi names cholesky-mode stochastic parameters 'sqrt(... stochastic ...)'.
        cls.is_stochastic = np.array(['stochastic' in str(lbl)
                                      for lbl in cls.target.parameter_labels])

    def column_norms(self, model):
        jac, _ = bd.jacobian_dict_to_array(model.sim.bulk_dprobs(self.circuits))
        return np.linalg.norm(jac, axis=0)


class PerturbErrorgenRatesTester(_ModelFixture, BaseCase):
    def test_the_target_model_has_invisible_stochastic_columns(self):
        """The reason perturb_errorgen_rates exists, pinned as a fact about pyGSTi.

        H+S stochastic rates use param_mode='cholesky': rate = theta**2, so
        d(rate)/d(theta) = 2*theta is exactly zero at the target model. A
        selector handed a target model optimises as if half the parameters were
        not there, silently.
        """
        self.assertGreater(self.is_stochastic.sum(), 0)
        norms = self.column_norms(self.target)
        self.assertLess(norms[self.is_stochastic].max(), 1e-4)
        self.assertGreater(np.median(norms[~self.is_stochastic]), 1.0)

    def test_perturbing_the_parameter_vector_is_not_enough(self):
        """The obvious fix does not work, which is why the helper sets rates.

        A 1e-4 nudge of the parameter vector puts theta at 1e-4, so the
        stochastic columns land at ~1e-4 of their rate-derivative: nonzero, but
        orders of magnitude under the unit ridge, so still invisible to the
        objective.
        """
        nudged = self.target.copy()
        rng = np.random.default_rng(0)
        nudged.from_vector(nudged.to_vector() + 1e-4 * rng.standard_normal(nudged.num_params))
        norms = self.column_norms(nudged)
        self.assertLess(norms[self.is_stochastic].max(), 1e-2)

    def test_perturbing_rates_makes_the_stochastic_columns_usable(self):
        perturbed = bd.perturb_errorgen_rates(self.target, 1e-3, seed=0)
        norms = self.column_norms(perturbed)
        hamiltonian = np.median(norms[~self.is_stochastic])
        stochastic = np.median(norms[self.is_stochastic])
        # Within an order of magnitude of the Hamiltonian columns, and well above
        # the unit ridge, is what "visible to the objective" means here.
        self.assertGreater(stochastic, 0.1)
        self.assertGreater(stochastic, hamiltonian / 100)

    def test_the_input_model_is_not_modified(self):
        before = self.target.to_vector().copy()
        bd.perturb_errorgen_rates(self.target, 1e-3, seed=0)
        self.assertArraysEqual(self.target.to_vector(), before)

    def test_the_seed_controls_the_result(self):
        a = bd.perturb_errorgen_rates(self.target, 1e-3, seed=0)
        b = bd.perturb_errorgen_rates(self.target, 1e-3, seed=0)
        c = bd.perturb_errorgen_rates(self.target, 1e-3, seed=1)
        self.assertArraysEqual(a.to_vector(), b.to_vector())
        self.assertFalse(np.allclose(a.to_vector(), c.to_vector()))

    def test_the_scale_is_a_rate_not_a_parameter_value(self):
        # rate = theta**2 in cholesky mode, so scaling rates by 100 scales the
        # stochastic *parameters* by 10. Pins that `scale` means what it says.
        small = bd.perturb_errorgen_rates(self.target, 1e-4, seed=0)
        large = bd.perturb_errorgen_rates(self.target, 1e-2, seed=0)
        ratio = (large.to_vector()[self.is_stochastic]
                 / small.to_vector()[self.is_stochastic])
        self.assertArraysAlmostEqual(ratio, np.full(ratio.shape, 10.0), places=6)

    def test_cptplnd_factory_preserves_physicality_and_full_sensitivity(self):
        from pygsti.modelpacks import smq1Q_XYI
        from pygsti.tools.jamiolkowski import fast_jamiolkowski_iso_std

        target = smq1Q_XYI.target_model('CPTPLND')
        before = target.to_vector().copy()
        samples = []
        for seed in (0, 1):
            with self.subTest(seed=seed):
                model = bd.BlockDoptReducer.from_target_model(target, seed=seed).model
                repeated = bd.perturb_errorgen_rates(target, seed=seed)
                self.assertArraysEqual(model.to_vector(), repeated.to_vector())
                self.assertEqual(model.num_params, target.num_params)
                self.assertArraysEqual(model.parameter_labels, target.parameter_labels)
                for label, op in model.operations.items():
                    choi = fast_jamiolkowski_iso_std(op.to_dense(), model.basis)
                    self.assertGreaterEqual(np.linalg.eigvalsh(choi).min(), -1e-12, label)
                    self.assertArraysAlmostEqual(op.to_dense()[0], [1., 0., 0., 0.], places=12)
                    # A diagonal noise sample must still expose all 12 gate
                    # parameters, including off-diagonal Cholesky directions.
                    singular_values = np.linalg.svd(op.deriv_wrt_params(), compute_uv=False)
                    self.assertEqual(len(singular_values), 12)
                    self.assertGreater(singular_values.min(), 1e-5, label)
                samples.append(model.to_vector().copy())
        self.assertFalse(np.allclose(samples[0], samples[1]))
        self.assertArraysEqual(target.to_vector(), before)

    def test_cptplnd_scale_changes_coefficients_linearly(self):
        from pygsti.modelpacks import smq1Q_XYI

        target = smq1Q_XYI.target_model('CPTPLND')
        small = bd.perturb_errorgen_rates(target, scale=1e-4, seed=0)
        large = bd.perturb_errorgen_rates(target, scale=1e-2, seed=0)
        for label in target.operations:
            small_coeffs = small.operations[label].errorgen_coefficients()
            large_coeffs = large.operations[label].errorgen_coefficients()
            self.assertArraysAlmostEqual(
                np.array(list(large_coeffs.values())),
                100 * np.array(list(small_coeffs.values())), places=12)
            stochastic = [value for key, value in small_coeffs.items() if key.errorgen_type == 'S']
            self.assertTrue(all(0 < value <= 1e-4 for value in stochastic))

    def test_depolarizing_parameterizations_preserve_tied_rates(self):
        from pygsti.modelpacks import smq1Q_XYI

        for parameterization in ('H+D', 'D'):
            with self.subTest(parameterization=parameterization):
                target = smq1Q_XYI.target_model(parameterization)
                model = bd.perturb_errorgen_rates(target, seed=0)
                self.assertArraysEqual(model.parameter_labels, target.parameter_labels)
                for label, op in model.operations.items():
                    stochastic = [value for key, value in op.errorgen_coefficients().items()
                                  if key.errorgen_type == 'S']
                    self.assertEqual(len(stochastic), 3)
                    self.assertTrue(all(value == stochastic[0] for value in stochastic), label)
                    self.assertGreater(stochastic[0], 0)

    def test_nonpositive_or_nonfinite_scale_is_rejected(self):
        for scale in (0, -1e-3, np.nan, np.inf, -np.inf):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, 'scale'):
                    bd.perturb_errorgen_rates(self.target, scale=scale, seed=0)


class RankCircuitsTester(_ModelFixture, BaseCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = bd.perturb_errorgen_rates(cls.target, 1e-3, seed=0)

    def _ranked(self, circuits, max_circuits=None, **kwargs):
        ranked, scores = bd.rank_circuits_by_dopt(self.model, circuits, max_circuits, **kwargs)
        self.assertEqual(len(scores), len(ranked))
        return ranked

    def test_ranks_every_unique_circuit_by_default(self):
        ranked = self._ranked(self.circuits)
        self.assertEqual(len(ranked), len(self.circuits))
        self.assertEqual(set(ranked), set(self.circuits))

    def test_a_budget_is_a_prefix_of_the_full_ranking(self):
        full = self._ranked(self.circuits)
        self.assertEqual(self._ranked(self.circuits, 5), full[:5])

    def test_is_deterministic(self):
        self.assertEqual(self._ranked(self.circuits, 6), self._ranked(self.circuits, 6))

    def test_duplicates_are_ranked_once(self):
        ranked = self._ranked(self.circuits + self.circuits)
        self.assertEqual(len(ranked), len(set(self.circuits)))
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_empty_candidates_and_zero_budget_do_not_simulate(self):
        with unittest.mock.patch.object(self.model.sim, 'bulk_dprobs',
                                        side_effect=AssertionError('No derivatives needed')):
            for circuits, budget in (([], None), ([], 3), (self.circuits, 0)):
                with self.subTest(circuits=len(circuits), budget=budget):
                    ranked, scores = bd.rank_circuits_by_dopt(self.model, circuits, budget)
                    self.assertEqual(ranked, [])
                    self.assertEqual(scores.shape, (0,))

    def test_scores_are_the_log_volume_curve_of_the_ranking(self):
        """Cross-checked against the independent Cholesky scorer, on the same matrix."""
        ranked, scores = bd.rank_circuits_by_dopt(self.model, self.circuits, 8)
        jac_dict = self.model.sim.bulk_dprobs(list(dict.fromkeys(self.circuits)))
        jac, block_size = bd.jacobian_dict_to_array(jac_dict)
        keys = list(jac_dict)
        pivots = [keys.index(c) for c in ranked]
        expected = bd.greedy_path_log_volumes(jac.T, block_size, pivots)
        self.assertArraysAlmostEqual(scores, expected[1:], places=8)

    def test_ridge_scales_the_objective(self):
        # A bigger ridge scales down each circuit's contribution to the objective;
        # the reported curve is 0.5*logdet(I + J^T J / ridge).
        _, unit = bd.rank_circuits_by_dopt(self.model, self.circuits, 5)
        _, heavy = bd.rank_circuits_by_dopt(self.model, self.circuits, 5, ridge=1e4)
        self.assertTrue(np.all(heavy < unit))

    def test_a_nonpositive_ridge_raises(self):
        for ridge in (0.0, -1.0):
            with self.subTest(ridge=ridge):
                with self.assertRaises(ValueError):
                    bd.rank_circuits_by_dopt(self.model, self.circuits, 2, ridge=ridge)

    def test_float32_ranks_nearly_the_same_circuits(self):
        # Not identical -- float32 reorders near-ties -- but the chosen *set* at a
        # generous budget should not move much, or the precision is not usable.
        f64 = set(self._ranked(self.circuits, 12))
        f32 = set(self._ranked(self.circuits, 12, dtype=np.float32))
        self.assertGreaterEqual(len(f64 & f32), 10)


class ReduceDesignTester(_ModelFixture, BaseCase):
    """`reduce_design_by_dopt` is duck-typed; this covers it on a plain design.

    The SimultaneousGSTDesign path is covered in
    test/unit/protocols/test_simultaneous_gst.py, where the fixture already exists.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = bd.perturb_errorgen_rates(cls.target, 1e-3, seed=0)

    def _design(self):
        from pygsti.protocols import CircuitListsDesign
        half = len(self.circuits) // 2
        return CircuitListsDesign([self.circuits[:half], self.circuits], nested=True)

    def test_keeps_the_budget_and_the_class(self):
        from pygsti.protocols import CircuitListsDesign
        design = self._design()
        reduced, _ = bd.reduce_design_by_dopt(design, self.model, 9)
        self.assertIsInstance(reduced, CircuitListsDesign)
        self.assertEqual(len(reduced.all_circuits_needing_data), 9)
        self.assertTrue(set(reduced.all_circuits_needing_data)
                        <= set(design.all_circuits_needing_data))

    def test_the_kept_circuits_are_the_ranking_prefix(self):
        design = self._design()
        reduced, _ = bd.reduce_design_by_dopt(design, self.model, 9)
        ranked, _ = bd.rank_circuits_by_dopt(self.model, design.all_circuits_needing_data, 9)
        self.assertEqual(set(reduced.all_circuits_needing_data), set(ranked))

    def test_the_original_design_is_not_modified(self):
        design = self._design()
        before = [list(cl) for cl in design.circuit_lists]
        bd.reduce_design_by_dopt(design, self.model, 5)
        self.assertEqual([list(cl) for cl in design.circuit_lists], before)

    def test_returns_the_score_curve_alongside_the_design(self):
        reduced, scores = bd.reduce_design_by_dopt(self._design(), self.model, 7)
        self.assertEqual(len(scores), 7)
        self.assertTrue(np.all(np.diff(scores) >= -1e-9))

    def test_a_budget_over_the_candidate_count_keeps_everything(self):
        design = self._design()
        reduced, _ = bd.reduce_design_by_dopt(design, self.model, 10 ** 6)
        self.assertEqual(set(reduced.all_circuits_needing_data),
                         set(design.all_circuits_needing_data))


class BlockDoptReducerTester(_ModelFixture, BaseCase):
    """The DesignReducer wrapper around the ranking.

    The selection algorithm is tested above; what matters here is that the object form
    agrees with the function form, carries the right diagnostics, and round-trips.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = bd.perturb_errorgen_rates(cls.target, 1e-3, seed=0)

    def _design(self):
        from pygsti.protocols import CircuitListsDesign
        half = len(self.circuits) // 2
        return CircuitListsDesign([self.circuits[:half], self.circuits], nested=True)

    # -- agreement with the function form ----------------------------------- #

    def test_it_selects_exactly_what_rank_circuits_by_dopt_ranks(self):
        """Pinned to the kernel, in order, so the wrapper cannot drift from it."""
        design = self._design()
        selection = bd.BlockDoptReducer(self.model).select(design, 9)
        ranked, scores = bd.rank_circuits_by_dopt(self.model, design.all_circuits_needing_data, 9)
        self.assertEqual(list(selection.circuits), list(ranked))
        self.assertArraysEqual(selection.scores, scores)

    def test_reduce_design_by_dopt_still_agrees_with_it(self):
        design = self._design()
        by_function, scores = bd.reduce_design_by_dopt(design, self.model, 9)
        by_object = bd.BlockDoptReducer(self.model).reduce(design, 9)
        self.assertEqual(set(by_function.all_circuits_needing_data),
                         set(by_object.all_circuits_needing_data))
        self.assertEqual(len(scores), 9)

    def test_function_reduction_records_and_replaces_gst_provenance(self):
        from pygsti.modelpacks import smq1Q_XYI
        design = smq1Q_XYI.create_gst_experiment_design(max_max_length=1)
        design = design.truncate_to_circuits(self.circuits[:12])
        by_function, scores = bd.reduce_design_by_dopt(design, self.model, 4)
        by_object = bd.BlockDoptReducer(self.model).reduce(design, 4)
        self.assertIsNotNone(by_function.selection)
        self.assertEqual(by_function.selection.circuits, by_object.selection.circuits)
        self.assertArraysEqual(by_function.selection.scores, scores)

        reduced, scores = bd.reduce_design_by_dopt(by_function, self.model, 2)
        self.assertEqual(set(reduced.selection.circuits), set(reduced.all_circuits_needing_data))
        self.assertEqual(len(reduced.selection.circuits), 2)
        self.assertArraysEqual(reduced.selection.scores, scores)

    def test_function_reduction_rejects_experiment_tree_children(self):
        from pygsti.protocols import CombinedExperimentDesign
        design = CombinedExperimentDesign({'child': self._design()})
        with self.assertRaisesRegex(NotImplementedError, 'child'):
            bd.reduce_design_by_dopt(design, self.model, 2)

    def test_an_empty_reduced_design_can_be_reduced_again(self):
        reducer = bd.BlockDoptReducer(self.model)
        empty = reducer.reduce(self._design(), 0)
        for budget in (None, 0, 3):
            with self.subTest(budget=budget):
                selection = reducer.select(empty, budget)
                self.assertEqual(selection.circuits, ())
                self.assertEqual(selection.scores.shape, (0,))
                self.assertEqual(selection.metadata['num_candidates'], 0)
                self.assertEqual(len(reducer.reduce(empty, budget).all_circuits_needing_data), 0)

    def test_selection_serializes_the_model_and_settings_used_for_the_selection(self):
        from pygsti.tools.edesigntools import CircuitSelection
        model = self.model.copy()
        before = model.to_vector().copy()
        reducer = bd.BlockDoptReducer(model, ridge=2.0)
        selection = reducer.select(self._design(), 2)
        reducer.ridge = 10.0
        changed = before.copy()
        changed[0] += 1e-4
        model.from_vector(changed)

        restored = CircuitSelection.from_nice_serialization(selection.to_nice_serialization())
        self.assertEqual(restored.reducer.ridge, 2.0)
        self.assertEqual(restored.metadata['ridge'], 2.0)
        self.assertArraysEqual(restored.reducer.model.to_vector(), before)

    def test_ridge_reaches_the_kernel(self):
        design = self._design()
        selection = bd.BlockDoptReducer(self.model, ridge=10.0).select(design, 6)
        ranked, _ = bd.rank_circuits_by_dopt(self.model, design.all_circuits_needing_data, 6,
                                               ridge=10.0)
        self.assertEqual(list(selection.circuits), list(ranked))

    def test_a_nonpositive_ridge_is_rejected_at_construction(self):
        for ridge in (0.0, -1.0):
            with self.subTest(ridge=ridge):
                with self.assertRaises(ValueError):
                    bd.BlockDoptReducer(self.model, ridge=ridge)

    # -- diagnostics --------------------------------------------------------- #

    def test_the_selection_carries_a_nondecreasing_score_curve(self):
        selection = bd.BlockDoptReducer(self.model).select(self._design(), 8)
        self.assertEqual(len(selection.scores), 8)
        self.assertTrue(np.all(np.diff(selection.scores) >= -1e-9))
        self.assertIn('logdet', selection.score_name)

    def test_the_selection_records_the_settings_and_the_candidate_count(self):
        design = self._design()
        selection = bd.BlockDoptReducer(self.model, ridge=2.0).select(design, 5)
        self.assertEqual(selection.metadata['ridge'], 2.0)
        self.assertEqual(selection.metadata['dtype'], 'float64')
        self.assertEqual(selection.metadata['num_candidates'],
                         len(design.all_circuits_needing_data))

    def test_no_budget_ranks_everything_so_the_curve_can_pick_one(self):
        design = self._design()
        selection = bd.BlockDoptReducer(self.model).select(design)
        n = len(design.all_circuits_needing_data)
        self.assertEqual(len(selection.circuits), n)
        # ...and the prefix is the answer for a smaller budget, with no re-ranking.
        self.assertEqual(list(selection.circuits[:6]),
                         list(bd.BlockDoptReducer(self.model).select(design, 6).circuits))

    # -- the target-model guard ---------------------------------------------- #

    def test_a_target_model_warns_and_says_how_to_fix_it(self):
        with self.assertWarns(UserWarning) as ctx:
            bd.BlockDoptReducer(self.target)
        self.assertIn('from_target_model', str(ctx.warning))

    def test_a_perturbed_model_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            bd.BlockDoptReducer(self.model)

    def test_the_warning_can_be_turned_off(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            bd.BlockDoptReducer(self.target, warn_on_target_model=False)

    def test_a_model_with_no_error_generators_does_not_warn(self):
        """Nothing to perturb means nothing to warn about; the check must not fire."""
        from pygsti.modelpacks import smq1Q_XYI
        full = smq1Q_XYI.target_model('full TP')
        self.assertFalse(bd._looks_like_a_target_model(full))

    def test_from_target_model_perturbs_and_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            reducer = bd.BlockDoptReducer.from_target_model(self.target, seed=0)
        # Same bar as test_perturbing_rates_makes_the_stochastic_columns_usable: rates
        # are drawn uniformly from [0, scale), so individual columns can still be small.
        norms = self.column_norms(reducer.model)
        self.assertGreater(np.median(norms[self.is_stochastic]), 0.1)

    def test_from_target_model_leaves_the_target_alone(self):
        before = self.target.to_vector().copy()
        bd.BlockDoptReducer.from_target_model(self.target, seed=0)
        self.assertArraysAlmostEqual(self.target.to_vector(), before)

    def test_from_target_model_is_reproducible_given_a_seed(self):
        design = self._design()
        first = bd.BlockDoptReducer.from_target_model(self.target, seed=7).select(design, 6)
        second = bd.BlockDoptReducer.from_target_model(self.target, seed=7).select(design, 6)
        self.assertEqual(list(first.circuits), list(second.circuits))

    # -- serialization -------------------------------------------------------- #

    def test_it_round_trips_and_still_picks_the_same_circuits(self):
        from pygsti.tools.edesigntools import DesignReducer
        reducer = bd.BlockDoptReducer(self.model, ridge=3.0, dtype=np.float32)
        restored = DesignReducer.from_nice_serialization(reducer.to_nice_serialization())
        self.assertIsInstance(restored, bd.BlockDoptReducer)
        self.assertEqual(restored.ridge, 3.0)
        self.assertEqual(restored.dtype, np.dtype(np.float32))
        design = self._design()
        self.assertEqual(list(restored.select(design, 6).circuits),
                         list(reducer.select(design, 6).circuits))

    def test_reloading_does_not_re_warn_about_the_model(self):
        """The model was vetted when the original was built; warning again is noise."""
        from pygsti.tools.edesigntools import DesignReducer
        state = bd.BlockDoptReducer(self.target, warn_on_target_model=False).to_nice_serialization()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            DesignReducer.from_nice_serialization(state)

    def test_a_model_that_cannot_be_walked_does_not_break_construction(self):
        """The check only drives a warning, so it must never be the thing that raises.

        `Model._iter_parameterized_objs` is abstract, and a member's coefficients need
        not be readable, so an exception here is reachable from a custom Model.
        """
        class _Opaque:
            def _iter_parameterized_objs(self):
                raise NotImplementedError

        class _BadMember:
            def errorgen_coefficients(self):
                raise RuntimeError("no coefficients for you")

        class _HasBadMember:
            def _iter_parameterized_objs(self):
                yield ('Gx', _BadMember())

        for model in (_Opaque(), _HasBadMember()):
            with self.subTest(model=type(model).__name__):
                self.assertFalse(bd._looks_like_a_target_model(model))
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    bd.BlockDoptReducer(model)
