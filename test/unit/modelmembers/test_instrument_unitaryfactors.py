import numpy as np
from scipy.optimize import minimize

from pygsti.baseobjs import Basis
from pygsti.modelmembers.instruments import Instrument, infer_unitary_factors
from pygsti.tools import basistools as bt
from ..util import BaseCase

PP = Basis.cast('pp', 4)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
P0, P1 = np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)


def _superop(K, basis=PP):
    """rho -> K rho K^dagger in `basis`."""
    return bt.change_basis(np.kron(K, K.conj()), 'std', basis).real


def _sqrtm_psd(E):
    w, V = np.linalg.eigh(E)
    return (V * np.sqrt(w.clip(0, None))) @ V.conj().T


def _same_up_to_phase(A, B):
    k = np.argmax(np.abs(B))
    return np.allclose(A * (B.flat[k] / A.flat[k]), B, atol=1e-9)


def _unitary(angles):
    a, b, c = angles
    return np.array([[np.cos(a) * np.exp(1j * b), -np.sin(a) * np.exp(-1j * c)],
                     [np.sin(a) * np.exp(1j * c), np.cos(a) * np.exp(-1j * b)]])


class InferUnitaryFactorsTester(BaseCase):

    def test_one_kraus_member_is_reproduced(self):
        rng = np.random.default_rng(0)
        for d in (2, 4):
            basis = Basis.cast('pp', d**2)
            V, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
            G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            E = G @ G.conj().T
            E /= 1.1 * np.linalg.eigvalsh(E).max()
            member = _superop(V @ _sqrtm_psd(E), basis)
            (pairs, factors) = infer_unitary_factors({'k': member}, basis)
            effect, U = pairs['k']
            self.assertTrue(_same_up_to_phase(U, V))
            self.assertLess(factors['k'].residual, 1e-12)
            self.assertArraysAlmostEqual(bt.vec_to_stdmx(effect, basis, keep_complex=True), E)

    def test_projective_member_gets_the_identity(self):
        pairs, factors = infer_unitary_factors({'p0': _superop(P0), 'p1': _superop(P1)}, PP)
        for lbl in ('p0', 'p1'):
            self.assertArraysAlmostEqual(pairs[lbl][1], np.eye(2))
            self.assertEqual(factors[lbl].completion_dim, 1)
            self.assertEqual(factors[lbl].unresolved_dim, 0)
            self.assertLess(factors[lbl].residual, 1e-12)

    def test_measure_and_reset_gets_a_bit_flip(self):
        # rho -> |0><1| rho |1><0|: the identity says nothing about where |0> should go,
        # so the completion swaps it with the supported direction.
        pairs, factors = infer_unitary_factors({'p0': _superop(P0), 'p1': _superop(X @ P1)}, PP)
        self.assertArraysAlmostEqual(pairs['p1'][1], X)
        self.assertEqual(factors['p1'].unresolved_dim, 0)
        self.assertLess(factors['p1'].residual, 1e-12)

    def test_zero_member_gets_the_reference(self):
        pairs, factors = infer_unitary_factors({'z': np.zeros((4, 4)), 'one': np.eye(4)}, PP)
        self.assertArraysAlmostEqual(pairs['z'][1], np.eye(2))
        self.assertEqual(factors['z'].leading_multiplicity, 0)
        pairs, _ = infer_unitary_factors({'z': np.zeros((4, 4)), 'one': np.eye(4)}, PP, reference={'z': Z})
        self.assertArraysAlmostEqual(pairs['z'][1], Z)
        self.assertArraysAlmostEqual(pairs['one'][1], np.eye(2))

    def test_tied_leading_kraus_operators_prefer_the_identity(self):
        # Equal mixture of rho -> sqrt(E) rho sqrt(E) and its Z-conjugate, with E = I/2.
        sqrtE = np.eye(2) / np.sqrt(2)
        member = 0.5 * _superop(sqrtE) + 0.5 * _superop(Z @ sqrtE)
        pairs, factors = infer_unitary_factors({'k': member, 'rest': _superop(sqrtE)}, PP)
        self.assertEqual(factors['k'].leading_multiplicity, 2)
        self.assertArraysAlmostEqual(pairs['k'][1], np.eye(2))

    def test_reference_completes_undetermined_directions(self):
        # P0 fixes U only on |0>; a Z reference sets the phase on |1>.
        pairs, _ = infer_unitary_factors({'p0': _superop(P0), 'p1': _superop(P1)}, PP, reference=Z)
        self.assertArraysAlmostEqual(pairs['p0'][1], Z)

    def test_multi_kraus_members_keep_their_effects(self):
        rng = np.random.default_rng(1)
        Ks = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(4)]
        S = sum(K.conj().T @ K for K in Ks)
        W = np.linalg.inv(_sqrtm_psd(S))
        Ks = [K @ W for K in Ks]  # sum_k K^dagger K = I
        members = {'a': _superop(Ks[0]) + _superop(Ks[1]), 'b': _superop(Ks[2]) + _superop(Ks[3])}
        pairs, factors = infer_unitary_factors(members, PP)
        rebuilt = Instrument.from_effects(pairs, PP)
        for lbl in members:
            E = Ks[0].conj().T @ Ks[0] + Ks[1].conj().T @ Ks[1] if lbl == 'a' else \
                Ks[2].conj().T @ Ks[2] + Ks[3].conj().T @ Ks[3]
            self.assertArraysAlmostEqual(bt.vec_to_stdmx(pairs[lbl][0], PP, keep_complex=True), E)
            self.assertArraysAlmostEqual(rebuilt[lbl].to_dense()[0], members[lbl][0])  # same effect row
            self.assertGreater(factors[lbl].residual, 1e-3)

    def test_within_three_times_the_best_unitary(self):
        # Brute-force reference: the best Ad_U o RootConj(E) over all 2x2 unitaries.
        rng = np.random.default_rng(2)
        for _ in range(5):
            K1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            K2 = 0.5 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            scale = np.sqrt(1.01 * np.linalg.eigvalsh(K1.conj().T @ K1 + K2.conj().T @ K2).max())
            member = _superop(K1 / scale) + _superop(K2 / scale)
            pairs, factors = infer_unitary_factors({'m': member}, PP)
            sqrtE = _sqrtm_psd(bt.vec_to_stdmx(pairs['m'][0], PP, keep_complex=True))
            dist = lambda t: np.linalg.norm(member - _superop(_unitary(t) @ sqrtE))
            best = min(minimize(dist, rng.uniform(0, 3, 3), method='Nelder-Mead',
                                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 4000}).fun for _ in range(6))
            self.assertLessEqual(factors['m'].residual, 3 * best + 1e-9)

    def test_non_cp_member_raises(self):
        with self.assertRaisesRegex(ValueError, "member 'bad'.*not completely positive"):
            infer_unitary_factors({'bad': 1.5 * _superop(P0) - 0.5 * _superop(P1)}, PP)
