"""Tests for the greedy block D-optimal kernel in pygsti.tools.edesign.blockdopt.

The kernel picks candidates by comparing R-factor diagonals from a Householder
QR.  Everything here judges its choices with `greedy_candidate_scores` and
`greedy_path_log_volumes` instead, which get the same quantity from a float64
`slogdet` of the information matrix and share no arithmetic with the kernel.
A test that scored the kernel with the kernel's own numbers would pass on a
consistently wrong implementation.

Ported from `python-block-edesigns/tests/test_reference_impls.py`, minus the
cases that compare against that package's compiled C++ kernel.
"""
import unittest.mock

import numpy as np

from pygsti.tools.edesign import blockdopt as bd

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
        piv, scores = bd.block_linear_dopt(A, 4, 8, return_scores=True)
        curve = bd.greedy_path_log_volumes(A, 4, piv)
        self.assertArraysAlmostEqual(curve[1:], scores, places=8)
        self.assertEqual(curve[0], 0.0)
        # Adding blocks can only add information.
        self.assertTrue(np.all(np.diff(curve) >= -1e-12))

    def test_first_pick_is_the_standalone_argmax(self):
        A = _design(31, 15, 6, 20)
        piv = bd.block_linear_dopt(A, 6, 1)
        self.assertEqual(piv.tolist(), [int(np.argmax(bd.greedy_candidate_scores(A, 6)))])

    def test_selection_is_greedy_optimal_at_every_step(self):
        for seed, ncand, b, p, k in [(0, 20, 4, 12, 8),
                                     (1, 30, 6, 16, 10),
                                     (2, 25, 1, 10, 12),      # scalar blocks
                                     (3, 12, 8, 3, 6)]:       # block_size > num params
            with self.subTest(seed=seed):
                A = _design(seed, ncand, b, p)
                piv = bd.block_linear_dopt(A, b, k)
                self.assertEqual(len(piv), min(ncand, k))
                self.assert_tol_greedy_path(A, b, piv)

    def test_ill_conditioned_input_stays_greedy_optimal(self):
        """Polynomially decaying singular values make later gains tiny (near-ties)."""
        rng = np.random.default_rng(12)
        ncand, b, p = 24, 4, 12
        U, _, Vt = np.linalg.svd(rng.standard_normal((ncand * b, p)), full_matrices=False)
        J = (U * (1 + np.arange(p)) ** -1.5) @ Vt
        A = np.ascontiguousarray(J.T)
        piv = bd.block_linear_dopt(A, b, ncand)
        self.assert_tol_greedy_path(A, b, piv, rel_tol=1e-6)

    def test_float32_selection_is_greedy_optimal_to_float32_tolerance(self):
        A = _design(1, 30, 6, 16, dtype=np.float32)
        piv = bd.block_linear_dopt(A, 6, 10)
        self.assert_tol_greedy_path(A, 6, piv, rel_tol=1e-3)

    # -- determinism and purity -------------------------------------------- #

    def test_is_deterministic_and_does_not_modify_input(self):
        A = _design(32, 12, 3, 9)
        A0 = A.copy()
        p1 = bd.block_linear_dopt(A, 3, 5)
        p2 = bd.block_linear_dopt(A, 3, 5)
        self.assertArraysEqual(p1, p2)
        self.assertArraysEqual(A, A0)

    def test_noncontiguous_input_gives_the_same_answer_as_a_copy(self):
        base = _design(11, 40, 4, 12)
        A = base[:, ::2]                            # 20 candidates, non-contiguous
        self.assertFalse(A.flags.c_contiguous or A.flags.f_contiguous)
        self.assertArraysEqual(bd.block_linear_dopt(A, 4, 8),
                               bd.block_linear_dopt(np.ascontiguousarray(A), 4, 8))

    def test_fortran_order_input_gives_the_same_answer(self):
        c = _design(0, 20, 4, 12, order="C")
        f = np.asfortranarray(c)
        self.assertArraysEqual(bd.block_linear_dopt(c, 4, 8), bd.block_linear_dopt(f, 4, 8))

    def test_exact_duplicate_blocks_tie_break_to_the_lowest_index(self):
        """Bitwise-identical blocks tie exactly; argmax takes the first maximum."""
        b, ncand, p = 4, 16, 10
        A = _design(20, ncand, b, p)
        dups = [(2, 9), (5, 13)]
        for lo, hi in dups:
            A[:, hi * b:(hi + 1) * b] = A[:, lo * b:(lo + 1) * b]
        piv = bd.block_linear_dopt(A, b, ncand)
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
        batched = bd.block_linear_dopt(A, 5, 11, return_scores=True)
        with unittest.mock.patch.object(bd, '_SCIPY_QR_IS_BATCHED', False):
            looped = bd.block_linear_dopt(A, 5, 11, return_scores=True)
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
        piv = bd.block_linear_dopt(A, 2, 0)
        self.assertEqual(piv.shape, (0,))
        self.assertEqual(piv.dtype, np.int64)
        piv, scores = bd.block_linear_dopt(A, 2, 0, return_scores=True)
        self.assertEqual(piv.shape, (0,))
        self.assertEqual(scores.shape, (0,))

    def test_max_blocks_is_clamped_to_the_candidate_count(self):
        A = _design(41, 5, 3, 7)
        self.assertEqual(len(bd.block_linear_dopt(A, 3, 100)), 5)

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

    def test_all_nan_input_raises(self):
        A = _design(43, 6, 4, 10)
        A[0, :] = np.nan                     # every candidate becomes "singular"
        with self.assertRaises(RuntimeError):
            bd.block_linear_dopt(A, 4, 3)

    def test_partial_nan_blocks_are_skipped(self):
        """Only the NaN-contaminated candidates are unselectable; the rest proceed."""
        A = _design(44, 10, 3, 8)
        bad = {1, 6}
        for blk in bad:
            A[0, blk * 3] = np.nan
        piv = bd.block_linear_dopt(A, 3, 8)
        self.assertEqual(len(piv), 8)
        self.assertFalse(set(piv.tolist()) & bad)

    def test_the_ridge_keeps_rank_deficient_input_selectable(self):
        """A rank-deficient candidate is not a singular workspace.

        W_i = [A_i^T ; I], so even an all-zero block has R = I and scores 0.
        The kernel therefore never runs out of valid candidates on finite
        input; the RuntimeError path is reachable only via NaN/Inf.  Pinned
        because it is the opposite of what "no non-singular candidate remains"
        suggests on first reading.
        """
        A = np.zeros((4, 12))
        piv, scores = bd.block_linear_dopt(A, 3, 4, return_scores=True)
        self.assertEqual(piv.tolist(), [0, 1, 2, 3])
        self.assertArraysAlmostEqual(scores, np.zeros(4), places=12)

    def test_zero_blocks_are_never_selected_while_others_remain(self):
        A = _design(45, 6, 3, 5)
        A[:, 3:6] = 0.0                      # candidate 1 adds nothing
        piv = bd.block_linear_dopt(A, 3, 5)
        self.assertNotIn(1, piv.tolist())


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
        maximised."""
        A = _design(55, 40, 4, 10)
        k = 8
        piv = bd.block_linear_dopt(A, 4, k)
        greedy = bd.greedy_path_log_volumes(A, 4, piv)[-1]
        rng = np.random.default_rng(0)
        for trial in range(10):
            subset = rng.choice(40, size=k, replace=False)
            with self.subTest(trial=trial):
                self.assertGreater(greedy, bd.greedy_path_log_volumes(A, 4, subset)[-1])
