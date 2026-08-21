import numpy as np

import pygsti.tools.chartools as ct
from pygsti.tools.internalgates import standard_gatename_unitaries
from pygsti.tools.optools import unitary_to_pauligate
from ..util import BaseCase


def _std_ptm(name):
    return unitary_to_pauligate(standard_gatename_unitaries()[name])


class AbelianCharacterTester(BaseCase):

    def test_character_orthogonality(self):
        for orders in [4, 3, (2, 4)]:
            table = ct.abelian_characters(orders)
            n = table.shape[0]
            gram = table @ table.conj().T
            self.assertArraysAlmostEqual(gram, n * np.identity(n))

    def test_character_values_z4(self):
        # chi_1(m) = i^m for Z4
        for m in range(4):
            self.assertAlmostEqual(ct.abelian_character(4, 1, m), 1j ** m)

    def test_character_weights_conjugate(self):
        weights = ct.character_weights(4, 1, [0, 1, 2, 3])
        self.assertArraysAlmostEqual(weights, np.conj(np.array([1, 1j, -1, -1j])))

    def test_product_group_character(self):
        # chi_(1,1)((m1,m2)) = (-1)^m1 * i^m2 on Z2 x Z4
        val = ct.abelian_character((2, 4), (1, 1), (1, 3))
        self.assertAlmostEqual(val, (-1) ** 1 * 1j ** 3)


class GermGroupTester(BaseCase):

    def test_germ_group_orders(self):
        s_ptm = _std_ptm('Gzpi2')
        y_ptm = _std_ptm('Gypi2')
        self.assertEqual(ct.germ_group_order(s_ptm), 4)
        self.assertEqual(ct.germ_group_order(y_ptm), 4)
        # triangle germ: S then sqrt(Y) applied as a single step
        self.assertEqual(ct.germ_group_order(y_ptm @ s_ptm), 3)
        # Hadamard-type germ S^2 * sqrt(Y)
        self.assertEqual(ct.germ_group_order(s_ptm @ s_ptm @ y_ptm), 2)

    def test_germ_group_order_raises(self):
        rot = unitary_to_pauligate(np.array([[1, 0], [0, np.exp(0.1j)]]))
        with self.assertRaises(ValueError):
            ct.germ_group_order(rot, max_order=24)

    def test_fourier_operators_are_projectors(self):
        s_ptm = _std_ptm('Gzpi2')
        ops = [ct.fourier_operator(s_ptm, 4, j) for j in range(4)]
        total = np.zeros((4, 4), dtype=complex)
        for j, op_j in enumerate(ops):
            self.assertArraysAlmostEqual(op_j @ op_j, op_j)  # idempotent
            for jj, op_jj in enumerate(ops):
                if jj != j:
                    self.assertArraysAlmostEqual(op_j @ op_jj, np.zeros((4, 4)))
            total += op_j
        self.assertArraysAlmostEqual(total, np.identity(4))

    def test_trivial_fourier_operator_rank(self):
        # For the ideal S germ the trivial irrep appears twice (I and Z directions)
        s_ptm = _std_ptm('Gzpi2')
        pi0 = ct.fourier_operator(s_ptm, 4, 0)
        self.assertAlmostEqual(np.trace(pi0).real, 2.0)


class ProjectorEigenvalueMapTester(BaseCase):

    def test_phase_gain(self):
        # arg f(e^{i theta}) ~= (order-1)/2 * theta for small theta
        theta = 1e-4
        for order in (2, 3, 4):
            fval = ct.projector_eigenvalue_map(np.exp(1j * theta), order)
            self.assertAlmostEqual(np.angle(fval) / theta, (order - 1) / 2.0, places=4)

    def test_fixed_points(self):
        for order in (2, 3, 4):
            self.assertAlmostEqual(ct.projector_eigenvalue_map(1.0, order), 1.0)
            # a full ideal-phase rotation of another irrep maps to 0
            other = np.exp(2j * np.pi / order)
            self.assertAlmostEqual(ct.projector_eigenvalue_map(other, order), 0.0)

    def test_matches_fourier_operator_eigenvalues(self):
        # Depolarize-and-overrotate S: noise commutes with the germ, so the Fourier
        # operator's eigenvalue on the nontrivial branch equals f(deviation).
        lam, theta = 0.98, 0.01
        s_ideal = _std_ptm('Gzpi2')
        overrot = unitary_to_pauligate(np.diag([1.0, np.exp(1j * (np.pi / 2 + theta))]))
        noisy = np.diag([1.0, lam, lam, 1.0]) @ overrot
        pi1 = ct.fourier_operator(noisy, 4, 1)
        evals = np.linalg.eigvals(pi1)
        dominant = evals[np.argmax(np.abs(evals))]
        deviation = lam * np.exp(1j * theta)  # noisy eigenvalue = i * deviation
        self.assertAlmostEqual(dominant, ct.projector_eigenvalue_map(deviation, 4), places=10)
