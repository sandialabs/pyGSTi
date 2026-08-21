import numpy as np

import pygsti.algorithms.cgstfit as cf
import pygsti.tools.chartools as ct
from ..util import BaseCase


class RealDecayFitTester(BaseCase):

    def setUp(self):
        self.depths = np.array([0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96])
        self.true = {'lam': 0.98, 'B': 0.97, 'C': 0.52}

    def _synthetic(self, noise_std=0.0, seed=0):
        t = self.true
        zs = (t['B'] - t['C']) * t['lam'] ** self.depths + t['C']
        if noise_std > 0:
            zs = zs + np.random.RandomState(seed).normal(0, noise_std, len(zs))
        return zs

    def test_noiseless_recovery(self):
        fit = cf.fit_real_decay(self.depths, self._synthetic())
        self.assertTrue(fit['success'])
        for key, val in self.true.items():
            self.assertAlmostEqual(fit['estimates'][key], val, places=6)

    def test_noisy_recovery(self):
        fit = cf.fit_real_decay(self.depths, self._synthetic(noise_std=1e-3),
                                bootstrap_samples=50,
                                rand_state=np.random.RandomState(1))
        for key, val in self.true.items():
            self.assertLess(abs(fit['estimates'][key] - val), 0.01)
        self.assertIsNotNone(fit['bootstrap_stderrs'])
        self.assertGreater(fit['bootstrap_stderrs']['lam'], 0)


class ComplexDecayFitTester(BaseCase):

    def setUp(self):
        self.depths = np.array([0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96])
        self.true = {'A': 0.95, 'lam': 0.985, 'theta': 0.012, 'phi': 0.03}

    def _synthetic(self, noise_std=0.0, seed=0):
        t = self.true
        zs = t['A'] * t['lam'] ** self.depths \
            * np.exp(1j * (t['theta'] * self.depths + t['phi']))
        if noise_std > 0:
            rs = np.random.RandomState(seed)
            zs = zs + rs.normal(0, noise_std, len(zs)) + 1j * rs.normal(0, noise_std, len(zs))
        return zs

    def test_noiseless_recovery(self):
        fit = cf.fit_complex_decay(self.depths, self._synthetic())
        self.assertTrue(fit['success'])
        for key, val in self.true.items():
            self.assertAlmostEqual(fit['estimates'][key], val, places=6)

    def test_noisy_recovery(self):
        fit = cf.fit_complex_decay(self.depths, self._synthetic(noise_std=1e-3),
                                   bootstrap_samples=50,
                                   rand_state=np.random.RandomState(1))
        self.assertLess(abs(fit['estimates']['theta'] - self.true['theta']), 5e-4)
        self.assertLess(abs(fit['estimates']['lam'] - self.true['lam']), 5e-3)
        self.assertIsNotNone(fit['bootstrap_stderrs'])

    def test_negative_theta(self):
        self.true['theta'] = -0.012
        fit = cf.fit_complex_decay(self.depths, self._synthetic())
        self.assertAlmostEqual(fit['estimates']['theta'], -0.012, places=6)


class ProjectorInversionTester(BaseCase):

    def test_roundtrip(self):
        rs = np.random.RandomState(7)
        for order in (2, 3, 4):
            for _ in range(20):
                y = rs.uniform(0.9, 1.0) * np.exp(1j * rs.uniform(-0.1, 0.1))
                z = ct.projector_eigenvalue_map(y, order)
                y_rec = cf.invert_projector_eigenvalue(z, order)
                self.assertAlmostEqual(y_rec, y, places=9)

    def test_identity_fixed_point(self):
        self.assertAlmostEqual(cf.invert_projector_eigenvalue(1.0, 4), 1.0, places=9)
