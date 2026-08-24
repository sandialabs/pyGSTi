import numpy as np

import pygsti
from pygsti.circuits import Circuit
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import true_germ_eigenvalues
from pygsti.protocols.cpe import (CharacterPhaseEstimation, CharacterPhaseEstimationDesign,
                                  create_2q_diagonal_cpe_design, create_2q_izz_unitary,
                                  extract_izz_phase_deviations, izz_coherence_irreps,
                                  make_2q_diagonal_model)
from pygsti.tools import chartools
from ..util import BaseCase, with_temp_path

_KS, _ORDER = (1, 3, 4), 13
_DEVIATIONS = (0.012, -0.007, 0.004)  # (d_iz, d_zi, d_zz) injected below
_DEPOL = 0.01
# eigenphase deviations of the targeted coherence irreps (5, 7, 10):
_TRUE_IRREP_PHASES = {5: _DEVIATIONS[0] + _DEVIATIONS[2],   # +0.016
                      7: _DEVIATIONS[1] + _DEVIATIONS[2],   # -0.003
                      10: _DEVIATIONS[0] - _DEVIATIONS[2]}  # +0.008


def _simulate(model, edesign, shots, sample_error, seed=None):
    ds = pygsti.data.simulate_data(model, edesign.all_circuits_needing_data, shots,
                                   sample_error=sample_error, seed=seed)
    return ProtocolData(edesign, ds)


class GateChoiceTester(BaseCase):
    """The (1,3,4) mod 13 diagonal gate has the properties the CPE example relies on."""

    def test_ideal_ptm_has_group_order_13(self):
        mdl = make_2q_diagonal_model(ks=_KS, order=_ORDER)
        germ = Circuit([('Gd', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        # the ideal model was overwritten with the (zero-deviation) noisy PTM
        ptm = mdl.sim.product(germ)
        self.assertEqual(chartools.germ_group_order(ptm, max_order=30), _ORDER)

    def test_all_twelve_coherences_in_distinct_irreps(self):
        # perfect-difference-set property: each nontrivial Z_13 irrep is singly occupied
        mdl = make_2q_diagonal_model(ks=_KS, order=_ORDER, deviations=_DEVIATIONS, depol=_DEPOL)
        germ = Circuit([('Gd', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        evals = np.linalg.eigvals(mdl.sim.product(germ))
        nontrivial = sorted(e for e in evals if abs(np.angle(e)) > 1e-6)
        self.assertEqual(len(nontrivial), 12)
        irrep_labels = sorted(round(np.angle(e) / (2 * np.pi / _ORDER)) % _ORDER
                              for e in nontrivial)
        self.assertEqual(irrep_labels, list(range(1, 13)))

    def test_true_eigenvalues_match_injected_deviations(self):
        mdl = make_2q_diagonal_model(ks=_KS, order=_ORDER, deviations=_DEVIATIONS, depol=_DEPOL)
        germ = Circuit([('Gd', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        truth = true_germ_eigenvalues(mdl, germ, _ORDER)
        for irrep, phase in _TRUE_IRREP_PHASES.items():
            self.assertAlmostEqual(np.angle(truth[irrep]), phase, places=12)
            self.assertAlmostEqual(abs(truth[irrep]), 1.0 - _DEPOL, places=12)

    def test_unitary_convention(self):
        # exp(-i pi/13 (IZ + 3 ZI + 4 ZZ)) has computational-basis phases
        # -(pi/13)*(8, -2, -6, 0) for |00>, |01>, |10>, |11>
        u = create_2q_izz_unitary(*(2 * np.pi * k / _ORDER for k in _KS))
        expected = np.exp(-1j * np.pi / _ORDER * np.array([8, -2, -6, 0]))
        self.assertArraysAlmostEqual(np.diag(u), expected)

    def test_izz_coherence_irreps(self):
        self.assertEqual(izz_coherence_irreps(_KS, _ORDER), (5, 7, 10))


class CharacterPhaseEstimationDesignTester(BaseCase):

    def test_construction(self):
        design = create_2q_diagonal_cpe_design(num_projection_rounds=2)
        self.assertEqual(design.mode, 'exact')
        self.assertEqual(design.irrep_indices, (5, 7, 10))
        self.assertEqual(design.circuits_per_depth, 2 * (_ORDER - 1) + 1)
        self.assertEqual(len(design.all_circuits_needing_data),
                         len(design.depths) * design.circuits_per_depth)

    def test_depth_validation(self):
        germ = Circuit([('Gd', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        with self.assertRaises(ValueError):  # missing the depth-0 phase reference
            CharacterPhaseEstimationDesign(germ, _ORDER, (5,), [1, 2, 4])
        with self.assertRaises(ValueError):  # missing the first generation
            CharacterPhaseEstimationDesign(germ, _ORDER, (5,), [0, 2, 4])
        with self.assertRaises(ValueError):  # more than doubles: 2 -> 8
            CharacterPhaseEstimationDesign(germ, _ORDER, (5,), [0, 1, 2, 8])
        with self.assertRaises(ValueError):  # trivial irrep carries no phase
            CharacterPhaseEstimationDesign(germ, _ORDER, (0,), [0, 1, 2, 4])

    def test_quadrature_weight_normalization(self):
        # per-irrep weights average any constant background to (near) zero
        design = create_2q_diagonal_cpe_design(num_projection_rounds=2)
        for j in design.irrep_indices:
            for weights in design.character_weights_for(j):
                self.assertLess(abs(np.mean(weights)), 1e-12)

    @with_temp_path
    def test_serialization_roundtrip(self, pth):
        design = create_2q_diagonal_cpe_design(depths=(0, 1, 2, 4), num_projection_rounds=2)
        design.write(pth)
        loaded = pygsti.io.read_edesign_from_dir(pth)
        self.assertEqual(tuple(loaded.irrep_indices), design.irrep_indices)
        self.assertEqual(loaded.group_order, design.group_order)
        self.assertEqual(loaded.mode, 'exact')
        self.assertEqual(list(loaded.all_circuits_needing_data),
                         list(design.all_circuits_needing_data))
        for j in design.irrep_indices:
            for w_orig, w_new in zip(design.character_weights_for(j),
                                     loaded.character_weights_for(j)):
                self.assertArraysAlmostEqual(w_orig, w_new)


class CharacterPhaseEstimationTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.edesign = create_2q_diagonal_cpe_design(depths=(0, 1, 2, 4, 8, 16, 32),
                                                    num_projection_rounds=2)
        cls.ideal_model = make_2q_diagonal_model(ks=_KS, order=_ORDER)
        cls.noisy_model = make_2q_diagonal_model(ks=_KS, order=_ORDER,
                                                 deviations=_DEVIATIONS, depol=_DEPOL)

    def test_ideal_model_gives_zero_phases(self):
        data = _simulate(self.ideal_model, self.edesign, 1000, 'none')
        results = CharacterPhaseEstimation(bootstrap_samples=0).run(data)
        for j, (phase, _) in results.phases_by_irrep().items():
            self.assertLess(abs(phase), 1e-6)

    def test_exact_probabilities_recover_injected_phases(self):
        # residual error is the k0=2 synthetic-projector bias, verified ~1e-5
        data = _simulate(self.noisy_model, self.edesign, 1000, 'none')
        results = CharacterPhaseEstimation(bootstrap_samples=0).run(data)
        for j, (phase, _) in results.phases_by_irrep().items():
            self.assertLess(abs(phase - _TRUE_IRREP_PHASES[j]), 2e-4)

    def test_shot_noise_recovery_with_bootstrap(self):
        data = _simulate(self.noisy_model, self.edesign, 2000, 'multinomial', seed=2026)
        results = CharacterPhaseEstimation(bootstrap_samples=50, seed=7).run(data)
        for j, (phase, stderr) in results.phases_by_irrep().items():
            self.assertIsNotNone(stderr)
            self.assertLess(abs(phase - _TRUE_IRREP_PHASES[j]), max(4 * stderr, 3e-3))

    def test_naive_unfiltered_signal_is_ambiguous(self):
        # Without character filtering, the |++> signal mixes all twelve
        # coherences (O(1) ideal frequencies), so the raw survival
        # probabilities carry no single-frequency phase near k*delta.
        data = _simulate(self.noisy_model, self.edesign, 1000, 'none')
        design = self.edesign
        depth_idx = design.depths.index(8)
        # the j=0 'exact'-mode circuit at each depth is the deterministic
        # prep * germ**k * meas circuit naive RPE would run
        circuits = design.circuit_lists[depth_idx]
        outs = design.idealout_lists[depth_idx]
        naive = [data.dataset[c].counts.get((out,), 0.0) / data.dataset[c].total
                 for c, draws, out in zip(circuits, design.exponent_lists[depth_idx], outs)
                 if draws == [0]]
        self.assertEqual(len(naive), 1)
        # interpreting 2p-1 as cos(8 * delta_5) is wrong by O(1)
        inferred = np.arccos(np.clip(2 * naive[0] - 1, -1, 1))
        self.assertGreater(abs(inferred - 8 * _TRUE_IRREP_PHASES[5]), 0.5)

    def test_extraction_inverts_deviation_map(self):
        # pure inversion identity, no data involved
        phases = {j: (p, 1e-4) for j, p in _TRUE_IRREP_PHASES.items()}
        out = extract_izz_phase_deviations(phases, ks=_KS, order=_ORDER)
        self.assertAlmostEqual(out['d_iz'], _DEVIATIONS[0], places=12)
        self.assertAlmostEqual(out['d_zi'], _DEVIATIONS[1], places=12)
        self.assertAlmostEqual(out['d_zz'], _DEVIATIONS[2], places=12)
        self.assertAlmostEqual(out['theta_iz'], 2 * np.pi * _KS[0] / _ORDER + _DEVIATIONS[0],
                               places=12)
        for key in ('d_iz_stderr', 'd_zi_stderr', 'd_zz_stderr'):
            self.assertGreater(out[key], 0.0)

    def test_end_to_end_extraction(self):
        data = _simulate(self.noisy_model, self.edesign, 2000, 'multinomial', seed=99)
        results = CharacterPhaseEstimation(bootstrap_samples=50, seed=11).run(data)
        out = extract_izz_phase_deviations(results.phases_by_irrep(), ks=_KS, order=_ORDER)
        for key, true_val in zip(('d_iz', 'd_zi', 'd_zz'), _DEVIATIONS):
            self.assertLess(abs(out[key] - true_val), max(4 * out[key + '_stderr'], 3e-3))
