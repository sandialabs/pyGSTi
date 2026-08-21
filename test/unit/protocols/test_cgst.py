import numpy as np
from scipy.linalg import expm

import pygsti
from pygsti.circuits import Circuit
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import (CharacterDecay, CharacterGST, CharacterGSTGermDesign,
                                   create_1q_szy_cgst_design, extract_szy_error_parameters,
                                   true_germ_eigenvalues)
from pygsti.tools.optools import unitary_to_pauligate as _u2p
from ..util import BaseCase, with_temp_path

_sx = np.array([[0, 1], [1, 0]], complex)
_sy = np.array([[0, -1j], [1j, 0]], complex)
_sz = np.array([[1, 0], [0, -1]], complex)


def _rot(axis, angle):
    """PTM of a rotation by `angle` about a Pauli axis."""
    return _u2p(expm(-0.5j * angle * axis))


def _pspec():
    return QubitProcessorSpec(1, ['Gzpi2', 'Gypi2', 'Gi'], qubit_labels=['Q0'])


def _standard_gauge_model(theta=0., alpha=0., beta=0., lam1=1., lam2=1., a=0.,
                          r1=0., r2=0., cxy=0., cxz=0., cyz=0., ay=0., arel=0.,
                          idle_angles=(0., 0., 0.)):
    """
    An ExplicitOpModel whose S and sqrt(Y) gates are EXACTLY the manuscript's
    standard-gauge channel forms (eq:S_Channel / eq:Y_Channel), so every
    standard-gauge error parameter has a known injected value.
    """
    E_S = np.array([[1, 0, 0, 0],
                    [0, lam2 * np.cos(theta), -lam2 * np.sin(theta), 0],
                    [0, lam2 * np.sin(theta), lam2 * np.cos(theta), 0],
                    [-a, 0, 0, lam1]])
    Lam_S = E_S @ _rot(_sz, np.pi / 2)
    E_sto = np.array([[1, 0, 0, 0],
                      [arel, 1 - r2, cxy, cxz],
                      [arel, cxy, 1 - r2, cyz],
                      [ay, cxz, cyz, 1 - r1]])
    Lam_Y = _rot(_sx, beta) @ E_sto @ _rot(_sy, np.pi / 2 + alpha) @ _rot(_sx, -beta)
    thx, thy, thz = idle_angles
    Lam_I = _rot(_sx, thx) @ _rot(_sy, thy) @ _rot(_sz, thz)

    mdl = create_explicit_model(_pspec(), ideal_gate_type='full')
    mdl.operations[('Gzpi2', 'Q0')] = Lam_S
    mdl.operations[('Gypi2', 'Q0')] = Lam_Y
    mdl.operations[('Gi', 'Q0')] = Lam_I
    return mdl


def _lindblad_noisy_model():
    """A generic Lindblad-parameterized noisy model (asymmetric stochastic noise)."""
    noise = {
        'Gzpi2:Q0': {('H', 'Z'): 0.005, ('S', 'X'): 0.004, ('S', 'Y'): 0.004, ('S', 'Z'): 0.008},
        'Gypi2:Q0': {('H', 'Y'): 0.004, ('H', 'X'): 0.003,
                     ('S', 'X'): 0.006, ('S', 'Y'): 0.003, ('S', 'Z'): 0.005},
        'Gi:Q0': {('H', 'Z'): 0.004, ('H', 'X'): 0.003,
                  ('S', 'X'): 0.002, ('S', 'Y'): 0.002, ('S', 'Z'): 0.002},
    }
    return create_explicit_model(_pspec(), lindblad_error_coeffs=noise)


def _run_all_decays(model, edesign, num_samples=1000, sample_error='none', seed=None,
                    **decay_kwargs):
    ds = pygsti.data.simulate_data(model, edesign.all_circuits_needing_data,
                                   num_samples, sample_error=sample_error, seed=seed)
    data = ProtocolData(edesign, ds)
    proto = CharacterDecay(**decay_kwargs)
    return {name: proto.run(data[name]) for name in edesign.keys()}


class CharacterGSTGermDesignTester(BaseCase):

    def setUp(self):
        self.germ = Circuit([('Gzpi2', 'Q0')], line_labels=('Q0',))
        self.depths = [0, 2, 5]

    def test_construction_reduced(self):
        design = CharacterGSTGermDesign(self.germ, 4, 1, self.depths, 8,
                                        mode='reduced', num_projection_rounds=3, seed=0)
        self.assertEqual(len(design.circuit_lists), len(self.depths))
        for k, circuits, exps, totals in zip(self.depths, design.circuit_lists,
                                             design.exponent_lists, design.total_germ_powers()):
            self.assertEqual(len(circuits), 8)
            for circ, draws, total in zip(circuits, exps, totals):
                self.assertEqual(len(draws), 3)
                self.assertEqual(total, k + sum(draws))
                self.assertEqual(len(circ), total)  # fiducials are empty here

    def test_construction_full(self):
        design = CharacterGSTGermDesign(self.germ, 4, 1, [1, 2, 4], 8, mode='full', seed=0)
        for k, exps, totals in zip([1, 2, 4], design.exponent_lists, design.total_germ_powers()):
            for draws, total in zip(exps, totals):
                self.assertEqual(len(draws), k)
                self.assertEqual(total, sum(draws))

    def test_full_mode_rejects_depth_zero(self):
        with self.assertRaises(ValueError):
            CharacterGSTGermDesign(self.germ, 4, 1, [0, 1], 8, mode='full')

    def test_residue_stratification(self):
        # with circuits_per_depth a multiple of the group order, the character
        # weights at each depth sum to zero exactly for nontrivial irreps
        design = CharacterGSTGermDesign(self.germ, 4, 1, self.depths, 8,
                                        mode='reduced', num_projection_rounds=4, seed=5)
        for weights in design.character_weights():
            self.assertAlmostEqual(abs(np.sum(weights)), 0.0, places=12)

    @with_temp_path
    def test_serialization_roundtrip(self, pth):
        design = create_1q_szy_cgst_design([0, 1, 4], 8, include_idle=True, seed=7)
        design.write(pth)
        loaded = pygsti.io.read_edesign_from_dir(pth)
        self.assertEqual(set(loaded.keys()), set(design.keys()))
        for name in design.keys():
            orig, new = design[name], loaded[name]
            self.assertEqual(orig.exponent_lists, new.exponent_lists)
            self.assertEqual(orig.germ, new.germ)
            self.assertEqual(orig.group_order, new.group_order)
            self.assertEqual(orig.irrep_index, new.irrep_index)
            self.assertEqual(orig.mode, new.mode)
            self.assertEqual(list(orig.all_circuits_needing_data),
                             list(new.all_circuits_needing_data))
            for w_orig, w_new in zip(orig.character_weights(), new.character_weights()):
                self.assertArraysAlmostEqual(w_orig, w_new)

    def test_szy_design_fiducial_conventions(self):
        design = create_1q_szy_cgst_design([0, 1], 4, include_idle=False, seed=0)
        empty = Circuit((), line_labels=('Q0',))
        # T1 experiment of S and Ramsey experiment of sqrt(Y) probe axes
        # already reachable natively: fiducials should be empty
        self.assertEqual(design['s_t1'].prep_fiducial, empty)
        self.assertEqual(design['y_ramsey'].prep_fiducial, empty)
        # Ramsey experiment of S needs a basis change into the XY plane
        self.assertEqual(len(design['s_ramsey'].prep_fiducial), 1)


class CharacterDecayTester(BaseCase):

    def test_ideal_model_flat_decays(self):
        edesign = create_1q_szy_cgst_design([0, 1, 2, 4, 8, 16], 12,
                                            include_idle=True, seed=11)
        ideal = create_explicit_model(_pspec())
        results = _run_all_decays(ideal, edesign)
        for name, res in results.items():
            self.assertAlmostEqual(res.germ_eigenvalue_magnitude, 1.0, places=5,
                                   msg='nonunit ideal decay for %s' % name)
            self.assertAlmostEqual(res.germ_eigenvalue_phase, 0.0, places=5,
                                   msg='nonzero ideal phase for %s' % name)

    def test_fits_match_numeric_truth(self):
        edesign = create_1q_szy_cgst_design([0, 1, 2, 4, 8, 16, 24, 32, 48], 12,
                                            include_idle=True, seed=13)
        model = _lindblad_noisy_model()
        results = _run_all_decays(model, edesign)
        for name, res in results.items():
            design = edesign[name]
            truth = true_germ_eigenvalues(model, design.germ, design.group_order)
            true_dev = truth[design.irrep_index]
            self.assertLess(abs(res.germ_eigenvalue_magnitude - abs(true_dev)), 2e-3,
                            msg='magnitude mismatch for %s' % name)
            self.assertLess(abs(res.germ_eigenvalue_phase - np.angle(true_dev)), 5e-4,
                            msg='phase mismatch for %s' % name)

    def test_full_mode_inversion_consistency(self):
        model = _lindblad_noisy_model()
        reduced = create_1q_szy_cgst_design([0, 1, 2, 4, 8, 16, 24], 12,
                                            mode='reduced', include_idle=False, seed=17)
        full = create_1q_szy_cgst_design([1, 2, 4, 8, 12, 16], 64,
                                         mode='full', include_idle=False, seed=17)
        res_reduced = _run_all_decays(model, reduced)['s_ramsey']
        res_full = _run_all_decays(model, full)['s_ramsey']
        self.assertLess(abs(res_full.germ_eigenvalue_magnitude
                            - res_reduced.germ_eigenvalue_magnitude), 3e-3)
        self.assertLess(abs(res_full.germ_eigenvalue_phase
                            - res_reduced.germ_eigenvalue_phase), 1e-3)
        # without inversion, full mode's raw phase is amplified by (N-1)/2 = 1.5
        # (finite-sampling noise on the full-mode fit limits the tolerance here)
        res_raw = _run_all_decays(model, full, invert_full_mode=False)['s_ramsey']
        self.assertLess(abs(res_raw.germ_eigenvalue_phase
                            / res_reduced.germ_eigenvalue_phase - 1.5), 0.15)


class SZYExtractionTester(BaseCase):
    """
    These tests adjudicate, numerically, the manuscript's flagged sign/coefficient
    ambiguities (eq:Triangle_Equations and the idle phase differences): the
    formulas implemented in extract_szy_error_parameters must reproduce
    parameters injected via the manuscript's own standard-gauge channel forms.
    """

    injected = dict(theta=0.010, alpha=0.008, beta=0.006, lam1=1 - 0.020,
                    lam2=1 - 0.015, a=-0.004, r1=0.018, r2=0.012,
                    cxy=0.004, cxz=0.003, cyz=0.002, ay=-0.003, arel=-0.002,
                    idle_angles=(0.002, 0.003, 0.004))

    def _extract(self, model):
        # 'exact' quadrature mode: zero character-sampling error, so these
        # tests isolate the extraction formulas themselves
        edesign = create_1q_szy_cgst_design([0, 1, 2, 4, 8, 16, 24, 32, 48], 12,
                                            mode='exact', include_idle=True, seed=19)
        return extract_szy_error_parameters(_run_all_decays(model, edesign))

    def test_extraction_recovers_injected_parameters(self):
        inj = self.injected
        params = self._extract(_standard_gauge_model(**inj))
        first_order_tol = 5e-4  # extraction formulas are first-order in the error rates
        self.assertLess(abs(params['theta'] - inj['theta']), first_order_tol)
        self.assertLess(abs(params['alpha'] - inj['alpha']), first_order_tol)
        self.assertLess(abs(params['beta'] - inj['beta']), first_order_tol)
        self.assertLess(abs(params['lambda1'] - inj['lam1']), first_order_tol)
        self.assertLess(abs(params['lambda2'] - inj['lam2']), first_order_tol)
        self.assertLess(abs(params['a'] - inj['a']), first_order_tol)
        c_sum = inj['cxy'] + inj['cxz'] + inj['cyz']
        self.assertLess(abs(params['c_sum'] - c_sum), 1e-3)
        active_combo = -inj['a'] + inj['ay'] + 2 * inj['arel']
        self.assertLess(abs(params['active_combo'] - active_combo), 1e-3)
        for comp, angle in zip('xyz', inj['idle_angles']):
            self.assertLess(abs(params['theta_idle_' + comp] - angle), 1e-3,
                            msg='idle %s-angle mismatch' % comp)

    def test_triangle_equation_signs(self):
        # c-dominated model: the triangle eigenvalue SPLITTING equals c_sum
        # with coefficient +1 (adjudicating the manuscript's flagged 2/3 and
        # sign).  A little uniform depolarization keeps the model CP -- with
        # correlated errors alone the trivial branch GROWS (eigenvalue > 1)
        # and probabilities leave [0, 1].
        depol = dict(lam1=0.99, lam2=0.99, r1=0.01, r2=0.01)
        for c in (0.003, 0.006):
            params = self._extract(_standard_gauge_model(cxy=c, cxz=c, cyz=c, **depol))
            self.assertLess(abs(params['c_sum'] - 3 * c), 3e-4)
        # beta-only model: delta_omega = 2*beta/sqrt(3)
        params = self._extract(_standard_gauge_model(beta=0.008, **depol))
        self.assertLess(abs(params['beta'] - 0.008), 3e-4)

    def test_end_to_end_with_shot_noise(self):
        inj = self.injected
        model = _standard_gauge_model(**inj)
        edesign = create_1q_szy_cgst_design([0, 1, 2, 4, 8, 16, 24, 32, 48, 64], 24,
                                            include_idle=True, seed=23)
        ds = pygsti.data.simulate_data(model, edesign.all_circuits_needing_data,
                                       1000, sample_error='multinomial', seed=2026)
        data = ProtocolData(edesign, ds)
        results = CharacterGST(bootstrap_samples=20, gateset_inversion='szy',
                               seed=4).run(data)
        top = results.for_protocol['CharacterGST']
        params = top.error_parameters
        self.assertLess(abs(params['theta'] - inj['theta']), 2e-3)
        self.assertLess(abs(params['alpha'] - inj['alpha']), 2e-3)
        self.assertLess(abs(params['beta'] - inj['beta']), 3e-3)
        self.assertLess(abs(params['lambda1'] - inj['lam1']), 3e-3)
        self.assertLess(abs(params['lambda2'] - inj['lam2']), 3e-3)
        # summaries and children exist for every sub-experiment
        self.assertEqual(set(top.decay_summaries.keys()), set(edesign.keys()))
        df = top.to_dataframe()
        self.assertGreater(len(df), len(edesign.keys()))
