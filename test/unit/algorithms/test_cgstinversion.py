import warnings

import numpy as np
from scipy.linalg import expm

from pygsti.algorithms import cgstinversion as ci
from pygsti.circuits import Circuit
from pygsti.data import simulate_data
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import (CharacterDecay, CharacterGSTDesign, CharacterGSTGermDesign,
                                   _select_fiducials, create_1q_szy_cgst_design,
                                   true_germ_eigenvalues)
from pygsti.tools.basistools import change_basis
from pygsti.tools.lindbladtools import create_elementary_errorgen
from ..util import BaseCase

# The depth range and finite-difference step below are coupled: a trivial-irrep
# ("T1") decay rate is only resolvable from the fit's competing asymptote when
# `(1 - lam) * max(depths)` is comfortably above the fitter's noise floor.  With
# `step = 1e-3` and `max(depths) = 128` the perturbed decays lose ~10% over the
# depth range, which is plenty; shortening the depth range to 32 degrades the
# finite differences by ~2 orders of magnitude.
DEPTHS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
STEP = 1e-3

_cache = {}


def _q(*gate_names):
    return Circuit([(n, 'Q0') for n in gate_names], line_labels=('Q0',))


def _target_model():
    if 'target' not in _cache:
        pspec = QubitProcessorSpec(1, ['Gzpi2', 'Gypi2'], qubit_labels=['Q0'])
        _cache['target'] = create_explicit_model(pspec, ideal_gate_type='full TP',
                                                 ideal_spam_type='full TP', simulator='matrix')
    return _cache['target']


def _szy_design():
    if 'szy' not in _cache:
        _cache['szy'] = create_1q_szy_cgst_design(DEPTHS, 12, mode='exact',
                                                  num_projection_rounds=3, include_idle=False)
    return _cache['szy']


def _extended_design():
    """The {S, sqrt(Y)} design plus the order-3 germ `S Y Y Y` and the order-2 germ `S S Y Y Y S Y`."""
    if 'ext' not in _cache:
        base = _szy_design()
        children = {name: base[name] for name in base.keys()}
        for name, germ, order, irrep in [
                ('g4_t1', _q('Gzpi2', 'Gypi2', 'Gypi2', 'Gypi2'), 3, 0),
                ('g4_ramsey', _q('Gzpi2', 'Gypi2', 'Gypi2', 'Gypi2'), 3, 1),
                ('g7_t1', _q('Gzpi2', 'Gzpi2', 'Gypi2', 'Gypi2', 'Gypi2', 'Gzpi2', 'Gypi2'), 2, 0)]:
            prep, meas = _select_fiducials(germ, order, irrep, ('Gzpi2', 'Gypi2'), 'Q0')
            children[name] = CharacterGSTGermDesign(
                germ, order, irrep, DEPTHS, 12, prep_fiducial=prep, meas_fiducial=meas,
                mode='exact', num_projection_rounds=3, qubit_labels=('Q0',))
        _cache['ext'] = CharacterGSTDesign(children, qubit_labels=('Q0',))
    return _cache['ext']


def _jacobian(key, design):
    if key not in _cache:
        _cache[key] = ci.first_order_design_matrix(_target_model(), design, step=STEP)
    return _cache[key]


def _random_coefficients(seed, rate, cp=False):
    """A random error generator coefficient vector of the given overall size."""
    labels = ci.errorgen_parameter_labels(_target_model())
    x = np.random.RandomState(seed).randn(len(labels))
    x /= np.linalg.norm(x)
    if cp:  # keep the stochastic rates positive & dominant so the model stays physical
        for i, (_, lbl) in enumerate(labels):
            if lbl.errorgen_type == 'S':
                x[i] = 3 * abs(x[i])
            elif lbl.errorgen_type in ('C', 'A'):
                x[i] *= 0.1
    return x * rate, labels


class ObservableLabelTester(BaseCase):

    def test_szy_labels(self):
        labels = ci.observable_labels(_szy_design(), _target_model())
        self.assertEqual(labels,
                         [('s_t1', 'log_magnitude'), ('s_t1', 'active'),
                          ('s_ramsey', 'log_magnitude'), ('s_ramsey', 'phase'),
                          ('y_t1', 'log_magnitude'), ('y_t1', 'active'),
                          ('y_ramsey', 'log_magnitude'), ('y_ramsey', 'phase'),
                          ('tri_t1', 'log_magnitude'), ('tri_t1', 'active'),
                          ('tri_ramsey', 'log_magnitude'), ('tri_ramsey', 'phase')])

    def test_extended_labels(self):
        labels = ci.observable_labels(_extended_design(), _target_model())
        self.assertEqual(len(labels), 18)
        # the order-2 germ's trivial block is 2-dimensional -> log_magnitude + active
        self.assertEqual([q for n, q in labels if n == 'g7_t1'], ['log_magnitude', 'active'])
        self.assertEqual([q for n, q in labels if n == 'g4_ramsey'], ['log_magnitude', 'phase'])

    def test_unsupported_child_is_skipped_with_warning(self):
        germ = _q('Gzpi2', 'Gzpi2', 'Gypi2', 'Gypi2', 'Gypi2', 'Gzpi2', 'Gypi2')
        prep, meas = _select_fiducials(germ, 2, 1, ('Gzpi2', 'Gypi2'), 'Q0')
        child = CharacterGSTGermDesign(germ, 2, 1, [0, 1, 2], 4, prep_fiducial=prep,
                                       meas_fiducial=meas, mode='exact',
                                       num_projection_rounds=2, qubit_labels=('Q0',))
        design = CharacterGSTDesign({'bad': child}, qubit_labels=('Q0',))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            labels = ci.observable_labels(design, _target_model())
        self.assertEqual(labels, [])
        self.assertEqual(len(caught), 1)
        self.assertIn('bad', str(caught[0].message))


class _FakeTrivialDecay(object):
    """Just enough of a CharacterDecayResults for the trivial-block fit helpers."""

    def __init__(self, depths, signal):
        self.depths = list(depths)
        self.signal = np.asarray(signal, dtype=complex)
        self.irrep_index = 0
        self.mode = 'reduced'
        from pygsti.algorithms import cgstfit
        self.fit = cgstfit.fit_real_decay(self.depths, self.signal.real)
        self.germ_eigenvalue_magnitude = float(self.fit['estimates']['lam'])
        self.germ_eigenvalue_phase = 0.0


class TrivialDecayRefitTester(BaseCase):

    def test_ripple_does_not_run_the_decay_rate_away(self):
        # A nearly affine curve with a depth-periodic ripple (what a 'reduced'-mode
        # synthetic projector leaves behind) used to be fit as a "spike":
        # lam -> 1e-14, absorbing the ripple.  The refit must now either fail
        # (falling back to the protocol's fit) or return a plausible rate, and the
        # observable must be finite and small either way.
        depths = np.array(DEPTHS, dtype=float)
        z = 0.5 - 1e-4 * depths + 3e-4 * (depths % 4 == 1) - 2e-4 * (depths % 4 == 2)
        res = _FakeTrivialDecay(DEPTHS, z)
        refit = ci._refit_trivial_decay(res)
        if refit is not None:
            self.assertGreater(refit[0], ci.TRIVIAL_LAMBDA_FLOOR)
        value = ci._quantity_value(res, 'log_magnitude')
        self.assertTrue(np.isfinite(value))
        self.assertLess(abs(value), 1e-2)
        self.assertTrue(np.isfinite(ci._quantity_value(res, 'active')))
        # stderr is linearized about the same point estimate as the value
        self.assertTrue(np.isfinite(ci._quantity_stderr(res, 'active')))

    def test_clean_decay_is_refit_accurately(self):
        depths = np.array(DEPTHS, dtype=float)
        lam, b, c = 0.9993, 0.5, 0.31
        res = _FakeTrivialDecay(DEPTHS, c + (b - c) * lam ** depths)
        self.assertFalse(ci._trivial_decay_is_degenerate(res))
        got = ci._refit_trivial_decay(res)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got[0], lam, places=9)
        self.assertAlmostEqual(ci._quantity_value(res, 'active'), (1 - lam) * (b - c), places=9)

    def test_exactly_affine_curve_is_degenerate(self):
        depths = np.array(DEPTHS, dtype=float)
        res = _FakeTrivialDecay(DEPTHS, 0.5 - 2e-4 * depths)
        self.assertTrue(ci._trivial_decay_is_degenerate(res))
        self.assertEqual(ci._quantity_value(res, 'log_magnitude'), 0.0)
        self.assertAlmostEqual(ci._quantity_value(res, 'active'), 2e-4, places=8)  # -slope

    def test_too_few_depths_warns(self):
        res = _FakeTrivialDecay([0, 4], [0.5, 0.4])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(ci._trivial_decay_is_degenerate(res))
        self.assertTrue(any('fewer than 3 depths' in str(w.message) for w in caught))

    def test_results_object_is_not_mutated(self):
        depths = np.array(DEPTHS, dtype=float)
        res = _FakeTrivialDecay(DEPTHS, 0.3 + 0.2 * 0.999 ** depths)
        before = set(vars(res).keys())
        cache = {}
        ci._quantity_value(res, 'log_magnitude', cache, 'x')
        ci._quantity_value(res, 'active', cache, 'x')
        self.assertEqual(set(vars(res).keys()), before)
        self.assertIn('x', cache)


class DesignModeTester(BaseCase):

    def _child(self, mode, depths):
        germ = _q('Gzpi2')
        prep, meas = _select_fiducials(germ, 4, 1, ('Gzpi2', 'Gypi2'), 'Q0')
        return CharacterGSTGermDesign(germ, 4, 1, depths, 4, prep_fiducial=prep, meas_fiducial=meas,
                                      mode=mode, num_projection_rounds=2, qubit_labels=('Q0',), seed=0)

    def test_full_mode_is_rejected(self):
        design = CharacterGSTDesign({'ok': self._child('exact', [0, 1, 2]),
                                     'full': self._child('full', [1, 2, 4])}, qubit_labels=('Q0',))
        with self.assertRaises(ValueError) as cm:
            ci.check_design_modes(design)
        self.assertIn("'full'", str(cm.exception))
        with self.assertRaises(ValueError):
            ci.first_order_design_matrix(_target_model(), design)

    def test_reduced_mode_warns(self):
        design = CharacterGSTDesign({'red': self._child('reduced', [0, 1, 2])}, qubit_labels=('Q0',))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ci.check_design_modes(design)
        self.assertTrue(any("'reduced'" in str(w.message) for w in caught))
        design = CharacterGSTDesign({'ex': self._child('exact', [0, 1, 2])}, qubit_labels=('Q0',))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ci.check_design_modes(design)
        self.assertEqual(len(caught), 0)


class ErrorgenModelTester(BaseCase):

    def test_parameter_labels(self):
        labels = ci.errorgen_parameter_labels(_target_model())
        self.assertEqual(len(labels), 24)  # 2 gates x (3 H + 3 S + 3 C + 3 A)
        gates = sorted(set(str(g) for g, _ in labels))
        self.assertEqual(gates, ['Gypi2:Q0', 'Gzpi2:Q0'])
        types = sorted(set(lbl.errorgen_type for _, lbl in labels))
        self.assertEqual(types, ['A', 'C', 'H', 'S'])

    def test_zero_coefficients_reproduce_target(self):
        target = _target_model()
        model = ci.model_from_errorgen_coefficients(target, {})
        for lbl, op in target.operations.items():
            self.assertArraysAlmostEqual(model.operations[lbl].to_dense(), op.to_dense())

    def test_coefficients_round_trip(self):
        target = _target_model()
        x, labels = _random_coefficients(1, 1e-2, cp=True)
        coeffs = ci.coefficients_to_dict(x, labels)
        model = ci.model_from_errorgen_coefficients(target, coeffs)
        for gate_lbl, per_gate in coeffs.items():
            got = model.operations[gate_lbl].errorgen_coefficients(return_basis=False,
                                                                   logscale_nonham=False)
            for elbl, value in per_gate.items():
                self.assertAlmostEqual(got[elbl], value, places=10)

    def test_convention_is_expm_of_generator_times_ideal(self):
        """A single elementary coefficient `c` must give `exp(c*L) @ G_ideal`."""
        target = _target_model()
        sz = np.array([[1, 0], [0, -1]], complex)  # unnormalized Pauli
        for typ, mats, coeff in [('H', (sz,), 0.037), ('S', (sz,), 0.021)]:
            gen = change_basis(create_elementary_errorgen(typ, *mats), 'std', 'pp').real
            labels = ci.errorgen_parameter_labels(target)
            (gate_lbl, elbl), = [(g, l) for g, l in labels
                                 if str(g) == 'Gzpi2:Q0' and l.errorgen_type == typ
                                 and l.basis_element_labels == ('Z',)]
            model = ci.model_from_errorgen_coefficients(target, {gate_lbl: {elbl: coeff}})
            expected = expm(coeff * gen) @ target.operations[gate_lbl].to_dense()
            self.assertArraysAlmostEqual(model.operations[gate_lbl].to_dense(), expected)

    def test_coefficients_to_dict(self):
        labels = ci.errorgen_parameter_labels(_target_model())
        x = np.arange(len(labels), dtype='d')
        d = ci.coefficients_to_dict(x, labels)
        self.assertEqual(len(d), 2)
        self.assertEqual(sum(len(v) for v in d.values()), len(labels))
        self.assertAlmostEqual(d[labels[5][0]][labels[5][1]], 5.0)


class SimulateObservablesTester(BaseCase):

    def test_ideal_model_gives_zero(self):
        y = ci.simulate_observables(_target_model(), _szy_design(), _target_model())
        self.assertEqual(len(y), 12)
        self.assertLess(np.abs(y).max(), 1e-9)

    def test_matches_fitted_germ_eigenvalues(self):
        target = _target_model()
        x, labels = _random_coefficients(4, 3e-3, cp=True)
        model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
        y = ci.simulate_observables(model, _szy_design(), target)
        obs = ci.observable_labels(_szy_design(), target)
        for germ, order, prefix in [(_q('Gzpi2'), 4, 's'), (_q('Gypi2'), 4, 'y'),
                                    (_q('Gzpi2', 'Gypi2'), 3, 'tri')]:
            true = true_germ_eigenvalues(model, germ, order)
            i_mag = obs.index((prefix + '_t1', 'log_magnitude'))
            self.assertAlmostEqual(y[i_mag], np.log(abs(true[0])), places=8)
            i_mag = obs.index((prefix + '_ramsey', 'log_magnitude'))
            i_ph = obs.index((prefix + '_ramsey', 'phase'))
            self.assertAlmostEqual(y[i_mag], np.log(abs(true[1])), places=8)
            self.assertAlmostEqual(y[i_ph], np.angle(true[1]), places=8)

    def test_agrees_with_observables_from_results(self):
        """The simulated and the fit-from-data observable paths must agree."""
        target = _target_model()
        design = _szy_design()
        x, labels = _random_coefficients(5, 2e-3, cp=True)
        model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
        ds = simulate_data(model, design.all_circuits_needing_data, num_samples=1000000,
                           sample_error='none', seed=20)
        results = {name: CharacterDecay(bootstrap_samples=0).run(ProtocolData(design[name], ds))
                   for name in design.keys()}
        y, stderr = ci.observables_from_results(results, design, target)
        y_sim = ci.simulate_observables(model, design, target)
        self.assertArraysAlmostEqual(y, y_sim, places=6)
        self.assertEqual(len(stderr), len(y))
        self.assertTrue(np.all(np.isfinite(stderr)))


class DesignMatrixTester(BaseCase):

    def test_szy_rank(self):
        jac = _jacobian('J_szy', _szy_design())
        self.assertEqual(jac.shape, (12, 24))
        svals = np.linalg.svd(jac, compute_uv=False)
        self.assertGreater(svals[10], 1e-2)          # 11 amplified directions
        self.assertLess(svals[11], 1e-6 * svals[0])  # ...and no twelfth
        self.assertEqual(np.sum(svals > 1e-6 * svals[0]), 11)

    def test_szy_default_rcond_inversion_rank(self):
        target, design = _target_model(), _szy_design()
        jac = _jacobian('J_szy', design)
        y_ideal = ci.simulate_observables(target, design, target)
        out = ci.invert_first_order(y_ideal, y_ideal, jac)  # default rcond=1e-8
        self.assertEqual(out['rank'], 11)
        self.assertEqual(out['unamplified_directions'].shape, (24, 13))

    def test_extended_design_rank(self):
        jac = _jacobian('J_ext', _extended_design())
        self.assertEqual(jac.shape, (18, 24))
        svals = np.linalg.svd(jac, compute_uv=False)
        self.assertEqual(np.sum(svals > 1e-4 * svals[0]), 13)
        self.assertGreater(svals[12], 1e-2)

    def test_first_order_accuracy_is_quadratic(self):
        target, design = _target_model(), _szy_design()
        jac = _jacobian('J_szy', design)
        obs = ci.observable_labels(design, target)
        errors = []
        for rate in (1e-3, 3e-4):
            x, labels = _random_coefficients(3, rate)
            model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
            y = ci.simulate_observables(model, design, target, obs_labels=obs)
            errors.append(np.abs(y - jac @ x).max())
            self.assertLess(errors[-1], 100 * rate ** 2)
        # a 3.33x smaller perturbation must shrink the residual by ~10x
        self.assertGreater(errors[0] / errors[1], 5.0)


class InversionTester(BaseCase):

    def test_recovers_amplified_projection(self):
        target, design = _target_model(), _extended_design()
        jac = _jacobian('J_ext', design)
        y_ideal = ci.simulate_observables(target, design, target)
        _, s, vt = np.linalg.svd(jac)
        rank = int(np.sum(s > 1e-4 * s[0]))
        proj = vt[:rank].T @ vt[:rank]

        rate = 1e-3
        x, labels = _random_coefficients(7, rate)
        model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
        y = ci.simulate_observables(model, design, target)
        out = ci.invert_first_order(y, y_ideal, jac, rcond=1e-4)

        self.assertEqual(out['rank'], rank)
        self.assertEqual(out['unamplified_directions'].shape, (24, 24 - rank))
        self.assertLess(np.abs(proj @ out['coefficients'] - proj @ x).max(), 100 * rate ** 2)
        self.assertLess(np.abs(out['residual']).max(), 100 * rate ** 2)
        # the min-norm solution has no component along the unamplified directions
        self.assertLess(np.abs(out['unamplified_directions'].T @ out['coefficients']).max(), 1e-10)

    def test_recovered_model_reproduces_germ_eigenvalues(self):
        target, design = _target_model(), _extended_design()
        jac = _jacobian('J_ext', design)
        y_ideal = ci.simulate_observables(target, design, target)
        rate = 1e-3
        x, labels = _random_coefficients(7, rate)
        model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
        y = ci.simulate_observables(model, design, target)
        out = ci.invert_first_order(y, y_ideal, jac, rcond=1e-4)
        estimated = ci.model_from_errorgen_coefficients(
            target, ci.coefficients_to_dict(out['coefficients'], labels))

        for germ, order in [(_q('Gzpi2'), 4), (_q('Gypi2'), 4), (_q('Gzpi2', 'Gypi2'), 3)]:
            true = true_germ_eigenvalues(model, germ, order)
            got = true_germ_eigenvalues(estimated, germ, order)
            self.assertEqual(set(true.keys()), set(got.keys()))
            for irrep, val in true.items():
                self.assertLess(abs(got[irrep] - val), 100 * rate ** 2)

    def test_weighted_inversion_reports_covariance(self):
        target, design = _target_model(), _szy_design()
        jac = _jacobian('J_szy', design)
        y_ideal = ci.simulate_observables(target, design, target)
        x, labels = _random_coefficients(9, 1e-3, cp=True)
        model = ci.model_from_errorgen_coefficients(target, ci.coefficients_to_dict(x, labels))
        y = ci.simulate_observables(model, design, target)

        stderr = np.full(len(y), 1e-4)
        stderr[0] = np.nan  # must be filled in from the median rather than blowing up
        out = ci.invert_first_order(y, y_ideal, jac, y_stderr=stderr, rcond=1e-4)
        self.assertIsNotNone(out['covariance'])
        self.assertEqual(out['covariance'].shape, (24, 24))
        self.assertTrue(np.all(np.isfinite(out['covariance'])))
        unweighted = ci.invert_first_order(y, y_ideal, jac, rcond=1e-4)
        self.assertArraysAlmostEqual(out['coefficients'], unweighted['coefficients'], places=6)
