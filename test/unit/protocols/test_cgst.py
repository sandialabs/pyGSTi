import warnings

import numpy as np
from scipy.linalg import expm

import pygsti
from pygsti.algorithms import cgstdesign, cgstgauge
from pygsti.baseobjs.label import Label
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
    An ExplicitOpModel whose S and sqrt(Y) gates are EXACTLY the standard-gauge
    channel forms (see :func:`extract_szy_error_parameters`), so every
    standard-gauge error parameter has a known injected value.
    """
    E_S = np.array([[1, 0, 0, 0],
                    [0, lam2 * np.cos(theta), -lam2 * np.sin(theta), 0],
                    [0, lam2 * np.sin(theta), lam2 * np.cos(theta), 0],
                    [-a, 0, 0, lam1]])
    Lam_S = E_S @ _rot(_sz, np.pi / 2)
    E_sto = np.array([[1, 0, 0, 0],
                      [arel, 1 - r2, cxy, cxz],
                      [ay, cxy, 1 - r1, cyz],
                      [arel, cxz, cyz, 1 - r2]])
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
    These tests pin down, numerically, the signs and coefficients of the
    closed-form {S, sqrt(Y)} extraction (the triangle equations and the idle
    phase differences): the formulas implemented in
    extract_szy_error_parameters must reproduce parameters injected via the
    standard-gauge channel forms.
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
        # with coefficient +1, because the triangle axis (x+y+z)/sqrt(3)
        # weights the three correlated rates equally (+2/3 c_sum on the trivial
        # branch, -1/3 c_sum on the complex ones).  A little uniform
        # depolarization keeps the model CP -- with
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


class LinearGatesetInversionTester(BaseCase):
    """
    End-to-end tests of `CharacterGST(gateset_inversion='linear')`: the generic
    (design-agnostic) first-order inversion of `pygsti.algorithms.cgstinversion`
    followed by the standard-gauge fixing of `pygsti.algorithms.cgstgauge`.

    The design matrix is the expensive part (~15 s); it depends only on the
    experiment design and the target model, so it is built once here and reused
    by every test through `CharacterGST`'s (class-level) cache.
    """

    # the trivial-irrep decays must bend measurably away from a straight line for
    # the finite-differenced design matrix to resolve them: see the notes on
    # `cgstinversion.first_order_design_matrix`
    depths = [0, 1, 2, 4, 8, 16, 32, 64, 128]

    truth_rates = {
        'Gzpi2:Q0': {('H', 'Z'): 0.0015, ('H', 'X'): 0.0006, ('S', 'X'): 0.0012,
                     ('S', 'Y'): 0.0012, ('S', 'Z'): 0.0015, ('C', 'X', 'Y'): 0.0004,
                     ('A', 'X', 'Z'): 0.0005},
        'Gypi2:Q0': {('H', 'Y'): 0.0012, ('H', 'X'): 0.0009, ('S', 'X'): 0.0015,
                     ('S', 'Y'): 0.0008, ('S', 'Z'): 0.0013, ('A', 'X', 'Y'): 0.0004},
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        pspec = QubitProcessorSpec(1, ['Gzpi2', 'Gypi2'], qubit_labels=['Q0'])
        cls.target = create_explicit_model(pspec, ideal_gate_type='full TP',
                                           ideal_spam_type='full TP', simulator='matrix')
        cls.truth = create_explicit_model(pspec, lindblad_error_coeffs=cls.truth_rates,
                                          lindblad_parameterization='GLND', simulator='matrix')
        # the amplificationally-complete germ set for {S, sqrt(Y)} has 5 germs of
        # orders (4, 4, 3, 3, 3), each with a 2-dimensional trivial block and a
        # multiplicity-one nontrivial irrep -> 10 children, all analyzable
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.design = cgstdesign.create_cgst_design(
                cls.target, cls.depths, 12, mode='exact', num_projection_rounds=3, seed=0)
        cls.truth_coefficients = cgstgauge.errorgen_coefficients_in_gauge(
            cgstgauge.fix_standard_gauge(cls.truth, cls.target, 'Gzpi2'), cls.target)

    def _protocol(self, **kwargs):
        kwargs.setdefault('bootstrap_samples', 0)
        return CharacterGST(gateset_inversion='linear', target_model=self.target,
                            reference_gate='Gzpi2', **kwargs)

    def _run(self, num_samples=1000, sample_error='none', data_seed=None, **kwargs):
        ds = pygsti.data.simulate_data(self.truth, self.design.all_circuits_needing_data,
                                       num_samples, sample_error=sample_error, seed=data_seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # the degenerate irrep is skipped noisily
            results = self._protocol(**kwargs).run(ProtocolData(self.design, ds))
        return results

    def _truth_entry(self, gate_str, errgen_str):
        gate = next(g for g in self.truth_coefficients if str(g) == gate_str)
        errgen = next(e for e in self.truth_coefficients[gate] if str(e) == errgen_str)
        return self.truth_coefficients[gate][errgen], errgen

    def test_design_structure(self):
        self.assertEqual(len(self.design.keys()), 10)
        self.assertEqual(len(set(row['germ'] for row in self.design.germ_table)), 5)
        self.assertTrue(all(self.design[name].mode == 'exact' for name in self.design.keys()))

    def test_noiseless_data_recovers_the_gate_set(self):
        top = self._run().for_protocol['CharacterGST']
        info = top.inversion_info
        self.assertEqual(info['rank'], 13)
        self.assertEqual(info['num_params'], 24)
        self.assertEqual(info['num_observables'], 20)
        self.assertEqual(info['num_unamplified'], 24 - 13)
        self.assertEqual(len(info['singular_values']), 20)
        self.assertLess(info['residual_norm'], 1e-2)
        self.assertEqual(info['skipped_children'], [])

        # (i) gauge-invariant check: every germ eigenvalue the design probes
        for row in self.design.germ_table:
            germ = Circuit(row['germ'])
            true = true_germ_eigenvalues(self.truth, germ, row['group_order'])
            got = true_germ_eigenvalues(top.estimated_model, germ, row['group_order'])
            for irrep, value in true.items():
                self.assertLess(abs(got[irrep] - value), 1e-4,
                                msg='germ eigenvalue mismatch for %s irrep %d'
                                    % (row['name'], irrep))

        # (ii) gauge-dependent check: the standard-gauge Hamiltonian coefficients
        for gate_str, per_gate in top.errorgen_estimates.items():
            for errgen_str, entry in per_gate.items():
                truth_value, errgen = self._truth_entry(gate_str, errgen_str)
                if errgen.errorgen_type != 'H':
                    continue
                self.assertLess(abs(entry['value'] - truth_value), 1e-4,
                                msg='%s %s: %g vs %g' % (gate_str, errgen_str,
                                                         entry['value'], truth_value))
        # ...and the full coefficient set is still first-order accurate
        self.assertEqual(sorted(top.errorgen_estimates.keys()), ['Gypi2:Q0', 'Gzpi2:Q0'])
        self.assertEqual(sum(len(v) for v in top.errorgen_estimates.values()), 24)

    def test_shot_noise_uncertainties(self):
        top = self._run(num_samples=2000, sample_error='multinomial', data_seed=2026,
                        bootstrap_samples=20, seed=7).for_protocol['CharacterGST']
        self.assertEqual(top.inversion_info['rank'], 13)
        for gate_str, per_gate in top.errorgen_estimates.items():
            for errgen_str, entry in per_gate.items():
                truth_value, _ = self._truth_entry(gate_str, errgen_str)
                stderr, deviation = entry['stderr'], abs(entry['value'] - truth_value)
                self.assertTrue(np.isfinite(stderr) and stderr >= 0,
                                msg='bad stderr for %s %s' % (gate_str, errgen_str))
                if abs(truth_value) > 1e-6:  # amplified, physically nonzero coefficients
                    self.assertGreater(stderr, 0.0)
                self.assertTrue(deviation < 5e-3 or deviation < 5 * stderr,
                                msg='%s %s off by %g (stderr %g)'
                                    % (gate_str, errgen_str, deviation, stderr))
        df = top.to_dataframe()
        errgen_rows = df[df['type'] == 'error generator']
        self.assertEqual(len(errgen_rows), 24)
        self.assertIn('Gzpi2:Q0:H(Z:Q0)', list(errgen_rows['quantity']))

    @with_temp_path
    def test_results_roundtrip(self, pth):
        results = self._run(num_samples=2000, sample_error='multinomial', data_seed=31,
                            bootstrap_samples=10, seed=3)
        results.write(pth)
        loaded = pygsti.io.read_results_from_dir(pth)
        top, reloaded = results.for_protocol['CharacterGST'], loaded.for_protocol['CharacterGST']
        self.assertEqual(reloaded.inversion_info, top.inversion_info)
        self.assertEqual(reloaded.errorgen_estimates, top.errorgen_estimates)
        self.assertEqual(set(loaded.keys()), set(results.keys()))
        for lbl, op in top.estimated_model.operations.items():
            self.assertArraysAlmostEqual(reloaded.estimated_model.operations[lbl].to_dense(),
                                         op.to_dense())
        self.assertEqual(reloaded.protocol.gateset_inversion, 'linear')
        self.assertEqual(reloaded.protocol.reference_gate, 'Gzpi2')
        for lbl, op in self.target.operations.items():
            self.assertArraysAlmostEqual(
                reloaded.protocol.target_model.operations[lbl].to_dense(), op.to_dense())

    @with_temp_path
    def test_szy_results_still_roundtrip_without_a_model(self, pth):
        edesign = create_1q_szy_cgst_design([0, 1, 2, 4], 12, mode='exact',
                                            include_idle=False, seed=5)
        ds = pygsti.data.simulate_data(_lindblad_noisy_model(),
                                       edesign.all_circuits_needing_data, 1000,
                                       sample_error='none')
        results = CharacterGST(bootstrap_samples=0, gateset_inversion='szy').run(
            ProtocolData(edesign, ds))
        top = results.for_protocol['CharacterGST']
        self.assertIsNotNone(top.error_parameters)
        self.assertIsNone(top.errorgen_estimates)
        self.assertIsNone(top.estimated_model)
        self.assertIsNone(top.inversion_info)
        # the new (unset) members must not break serialization
        results.write(pth)
        reloaded = pygsti.io.read_results_from_dir(pth).for_protocol['CharacterGST']
        self.assertEqual(reloaded.error_parameters, top.error_parameters)
        self.assertIsNone(reloaded.estimated_model)
        self.assertIsNone(reloaded.protocol.target_model)
        self.assertIsNone(reloaded.protocol.reference_gate)

    def test_linear_requires_target_model_and_reference_gate(self):
        with self.assertRaises(ValueError):
            CharacterGST(gateset_inversion='linear')
        with self.assertRaises(ValueError):
            CharacterGST(gateset_inversion='linear', target_model=self.target)
        with self.assertRaises(ValueError):
            CharacterGST(gateset_inversion='linear', reference_gate='Gzpi2')
        with self.assertRaises(ValueError):
            CharacterGST(gateset_inversion='bogus')
        # ...but the two supported inversions (and no inversion) are fine
        CharacterGST(gateset_inversion=None)
        CharacterGST(gateset_inversion='szy')
        CharacterGST(gateset_inversion='linear', target_model=self.target,
                     reference_gate=Label(('Gzpi2', 'Q0')), other_gates=['Gypi2'])
