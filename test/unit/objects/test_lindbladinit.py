import numpy as np
import pytest

from pygsti.baseobjs import Label as L
from pygsti.circuits import Circuit
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.models import fit_instrument_parameters, prepare_lindblad_seed
from pygsti.models.modelparaminterposer import LinearInterposer
from pygsti.tools import basistools as bt
from pygsti.tools import optools as ot
from pygsti.tools.jamiolkowski import jamiolkowski_iso
from pygsti.tools import sdptools as sdps
from ..util import BaseCase

BUDGET = 0.02


def _instrument_model(parameterization='CPTPLND'):
    """A 1-qubit Lindblad model plus an instrument built from ideal projective Z effects."""
    model = std.target_model()
    model.set_all_parameterizations(parameterization)
    E0 = np.diag([1.0, 0.0]).astype(complex)
    model[L('Iz', 0)] = Instrument.from_effects({'p0': E0, 'p1': np.eye(2) - E0}, model.basis)
    return model


def _errorgen(op):
    """The LindbladErrorgen of a ComposedOp([static, ExpErrorgenOp]) gate."""
    return op.factorops[1].errorgen


class PrepareLindbladSeedTester(BaseCase):

    def setUp(self):
        self.model = _instrument_model()
        self.vector, self.report = prepare_lindblad_seed(self.model, BUDGET, seed=3)
        self.prepared = self.model.copy()
        self.prepared.from_vector(self.vector)

    def test_every_degenerate_block_moves_to_its_interior(self):
        # rho0, Mdefault, the idle and two gates, plus the instrument's shared effect error map and
        # its two postgate error maps.
        self.assertEqual(len(self.report.perturbed), 8)
        self.assertEqual(self.report.skipped, ())
        # Pauli basis, so each S direction's diamond weight is 2; three directions per qubit.
        floor = 0.5 * 0.5 * BUDGET / 3 / 2
        for entry in self.report.perturbed:
            self.assertEqual(entry.min_dissipative_eigenvalue_before, 0.0)
            self.assertGreater(entry.min_dissipative_eigenvalue_after, 0.0)
            self.assertGreaterEqual(entry.min_stochastic_rate, floor * (1 - 1e-12))
            self.assertAlmostEqual(entry.diamond_bound, BUDGET, places=14)

    def test_shared_instrument_error_map_is_perturbed_once(self):
        shared = [e for e in self.report.perturbed if 'RootConjOperator' in e.paths[0]]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(shared[0].paths), 2)  # reached through both instrument members

    def test_effects_are_bounded_away_from_the_spectral_boundary(self):
        # In the unnormalized Pauli basis, lambda_min(E) >= exp(-sum s) * min(s) * d * Tr(E_static).
        povm_errgen = self.prepared[L('Iz', 0)]['p0'].factorops[0].submembers()[0].error_map.errorgen
        s = np.array([c.real for k, c in povm_errgen.coefficients(label_type='local').items()
                      if k.errorgen_type == 'S'])
        lower = np.exp(-s.sum()) * s.min() * 2 * 1.0
        self.assertEqual(len(self.report.effect_min_eigenvalues), 2)
        for lam in self.report.effect_min_eigenvalues.values():
            self.assertGreaterEqual(lam, lower)

    def test_removes_zero_jacobian_columns(self):
        circuits = list(std.create_gst_experiment_design(2).all_circuits_needing_data)
        circuits += [Circuit([L('Iz', 0)], line_labels=(0,)),
                     Circuit([L('Gxpi2', 0), L('Iz', 0), L('Gypi2', 0)], line_labels=(0,)),
                     Circuit([L('Gxpi2', 0), L('Iz', 0), L('Gxpi2', 0), L('Iz', 0)], line_labels=(0,))]

        def zero_columns(model):
            model.sim = 'matrix'
            dprobs = model.sim.bulk_dprobs(circuits)
            J = np.array([row for c in circuits for row in dprobs[c].values()])
            return np.count_nonzero(np.linalg.norm(J, axis=0) < 1e-12)

        self.assertGreater(zero_columns(self.model.copy()), 0)
        self.assertEqual(zero_columns(self.prepared), 0)

    def test_seed_reproduces_the_vector(self):
        again, _ = prepare_lindblad_seed(self.model, BUDGET, seed=3)
        self.assertArraysEqual(again, self.vector)
        fresh, report = prepare_lindblad_seed(self.model, BUDGET)
        replay, _ = prepare_lindblad_seed(self.model, BUDGET, seed=report.seed)
        self.assertArraysEqual(replay, fresh)

    def test_model_is_not_modified(self):
        self.assertArraysEqual(self.model.to_vector(), _instrument_model().to_vector())


class PrepareLindbladSeedNoisyModelTester(BaseCase):

    def setUp(self):
        self.model = std.target_model()
        self.model.set_all_parameterizations('CPTPLND')
        rng = np.random.default_rng(5)
        self.model.from_vector(self.model.to_vector() + 0.03 * rng.standard_normal(self.model.num_params))

    def test_existing_coefficients_are_kept(self):
        vector, report = prepare_lindblad_seed(self.model, BUDGET, seed=1)
        prepared = self.model.copy()
        prepared.from_vector(vector)
        before = _errorgen(self.model.operations[L('Gxpi2', 0)]).coefficients(label_type='local')
        after = _errorgen(prepared.operations[L('Gxpi2', 0)]).coefficients(label_type='local')
        for k in before:
            diff = after[k] - before[k]
            if k.errorgen_type in ('C', 'A'):
                self.assertAlmostEqual(abs(diff), 0.0, places=12)
            elif k.errorgen_type == 'S':
                self.assertGreater(diff.real, 0.0)
        for entry in report.perturbed:
            self.assertGreater(entry.min_dissipative_eigenvalue_after, entry.min_dissipative_eigenvalue_before)

    @pytest.mark.skipif(not sdps.CVXPY_ENABLED, reason="cvxpy is not installed.")
    def test_diamond_norm_change_is_within_budget(self):
        budget = 0.05
        vector, _ = prepare_lindblad_seed(self.model, budget, seed=2)
        prepared = self.model.copy()
        prepared.from_vector(vector)
        for lbl in self.model.operations:
            change = ot.diamonddist(self.model.operations[lbl].to_dense(), prepared.operations[lbl].to_dense(), 'pp')
            self.assertLessEqual(change, budget + 1e-6)


class PrepareLindbladSeedScopeTester(BaseCase):

    def test_skips_parameterizations_without_degenerate_blocks(self):
        model = std.target_model()
        model.set_all_parameterizations('GLND')
        vector, report = prepare_lindblad_seed(model, BUDGET, seed=1)
        self.assertEqual(report.perturbed, ())
        self.assertEqual(len(report.skipped), 5)
        self.assertArraysEqual(vector, model.to_vector())

    def test_depol_block_gets_a_common_rate(self):
        model = std.target_model()
        model.set_all_parameterizations('H+D')
        vector, report = prepare_lindblad_seed(model, BUDGET, seed=1)
        model.from_vector(vector)
        coeffs = _errorgen(model.operations[L('Gxpi2', 0)]).coefficients(label_type='local')
        s = [c.real for k, c in coeffs.items() if k.errorgen_type == 'S']
        self.assertArraysAlmostEqual(np.array(s), np.full(3, s[0]))
        self.assertGreater(s[0], 0.0)
        for entry in report.perturbed:
            self.assertAlmostEqual(entry.diamond_bound, BUDGET, places=14)

    def test_stochastic_only_split(self):
        model = std.target_model()
        model.set_all_parameterizations('CPTPLND')
        vector, report = prepare_lindblad_seed(model, BUDGET, hs_split=(0.0, 1.0), seed=1)
        model.from_vector(vector)
        coeffs = _errorgen(model.operations[L('Gxpi2', 0)]).coefficients(label_type='local')
        self.assertTrue(all(abs(c) == 0 for k, c in coeffs.items() if k.errorgen_type == 'H'))
        for entry in report.perturbed:
            self.assertAlmostEqual(entry.diamond_bound, BUDGET, places=14)

    def test_defaults(self):
        model = std.target_model()
        model.set_all_parameterizations('CPTPLND')
        _, report = prepare_lindblad_seed(model, seed=1)
        self.assertEqual((report.diamond_budget, report.hs_split, report.floor_fraction), (1e-3, (0.5, 0.5), 0.5))
        for entry in report.perturbed:
            self.assertAlmostEqual(entry.diamond_bound, 1e-3, places=14)

    def test_hamiltonian_share(self):
        model = std.target_model()
        model.set_all_parameterizations('CPTPLND')
        vector, report = prepare_lindblad_seed(model, BUDGET, hs_split=(0.5, 0.5), seed=1)
        model.from_vector(vector)
        coeffs = _errorgen(model.operations[L('Gxpi2', 0)]).coefficients(label_type='local')
        h = np.array([c.real for k, c in coeffs.items() if k.errorgen_type == 'H'])
        self.assertAlmostEqual(2 * np.abs(h).sum(), 0.5 * BUDGET, places=14)  # unit operator norms in PP

    def test_rejects_invalid_arguments(self):
        model = _instrument_model()
        for kwargs in ({'diamond_budget': 0.0}, {'diamond_budget': np.inf},
                       {'diamond_budget': BUDGET, 'hs_split': (0.5, 0.6)},
                       {'diamond_budget': BUDGET, 'hs_split': (1.0, 0.0)},
                       {'diamond_budget': BUDGET, 'floor_fraction': 1.5}):
            with self.assertRaises(ValueError):
                prepare_lindblad_seed(model, **kwargs)

    def test_rejects_parameter_interposer(self):
        model = std.target_model()
        model.set_all_parameterizations('CPTPLND')
        model.param_interposer = LinearInterposer(np.eye(model.num_params))
        with self.assertRaises(NotImplementedError):
            prepare_lindblad_seed(model, BUDGET, seed=1)

    def test_rejects_error_generators_sharing_parameters(self):
        model = std.target_model()
        model.set_all_parameterizations('CPTPLND')
        gx = int(model.operations[L('Gxpi2', 0)].gpindices_as_array()[0])
        gy = int(model.operations[L('Gypi2', 0)].gpindices_as_array()[0])
        model.collect_parameters([gx, gy])
        with self.assertRaisesRegex(ValueError, 'share model parameter'):
            prepare_lindblad_seed(model, BUDGET, seed=1)


class FitInstrumentParametersTester(BaseCase):

    def setUp(self):
        self.model = _instrument_model()
        self.label = L('Iz', 0)
        basis = self.model.basis
        # a weak measurement followed by a depolarizing kick, which the ideal-base chart can reach
        D = np.eye(4)
        D[1:, 1:] *= 0.98
        E0 = np.diag([0.92, 0.08]).astype(complex)
        sk = lambda m: bt.stdmx_to_vec(m, basis).ravel().real
        self.weak = {'p0': D @ ot.rootconj_superop(sk(E0), basis),
                     'p1': D @ ot.rootconj_superop(sk(np.eye(2) - E0), basis)}

    def _fitted(self, vector):
        fitted = self.model.copy()
        fitted.from_vector(vector)
        return fitted[self.label]

    def test_reaches_reachable_members(self):
        # the members of the same chart at random interior parameters, and the weak measurement
        vector, _ = prepare_lindblad_seed(self.model, BUDGET, seed=11)
        noisy = self.model.copy()
        noisy.from_vector(vector)
        for name, members in (('chart', noisy[self.label]), ('weak', self.weak)):
            with self.subTest(members=name):
                vector, fits = fit_instrument_parameters(self.model, {self.label: members}, tol=1e-9)
                fit = fits[self.label]
                self.assertTrue(fit.reached)
                self.assertLessEqual(fit.residual, 1e-9)
                self.assertGreater(fit.start_residual, 1e-2)
                inst = self._fitted(vector)
                for k, m in members.items():
                    target = m.to_dense() if hasattr(m, 'to_dense') else m
                    np.testing.assert_allclose(inst[k].to_dense(), target, atol=1e-9)

    def test_other_parameters_and_model_are_unchanged(self):
        before = self.model.to_vector().copy()
        vector, _ = fit_instrument_parameters(self.model, {self.label: self.weak})
        np.testing.assert_array_equal(self.model.to_vector(), before)
        others = np.setdiff1d(np.arange(len(before)), self.model[self.label].gpindices_as_array())
        np.testing.assert_array_equal(vector[others], before[others])

    def test_fit_that_stops_early_reports_it(self):
        _, fits = fit_instrument_parameters(self.model, {self.label: self.weak}, max_iter=1)
        fit = fits[self.label]
        self.assertFalse(fit.reached)
        self.assertGreater(fit.residual, fit.tol)
        self.assertIn('Maximum iterations', fit.message)

    def test_unreachable_members_end_on_a_cp_least_squares_fit(self):
        # Stretching the Bloch-vector block makes the members TP but not even positive.
        stretched = {k: m.copy() for k, m in self.weak.items()}
        for m in stretched.values():
            m[1:, 1:] *= 1.05
        vector, fits = fit_instrument_parameters(self.model, {self.label: stretched})
        fit = fits[self.label]
        self.assertFalse(fit.reached)
        self.assertIn('relative reductions', fit.message)
        self.assertLess(fit.residual, 0.5 * fit.start_residual)
        inst = self._fitted(vector)
        for k, m in inst.items():
            choi = jamiolkowski_iso(m.to_dense(), self.model.basis, 'std')
            self.assertGreater(np.linalg.eigvalsh(choi).min(), -1e-9)
        # the fit is (locally) least-squares: its residual is no worse than the weak members' own
        self.assertLess(fit.residual, np.sqrt(sum(np.linalg.norm(stretched[k] - self.weak[k])**2 for k in stretched)))

    def test_rejects_mismatched_members(self):
        with self.assertRaisesRegex(ValueError, 'no instrument'):
            fit_instrument_parameters(self.model, {L('Ix', 0): self.weak})
        with self.assertRaisesRegex(ValueError, 'outcomes'):
            fit_instrument_parameters(self.model, {self.label: {'p0': self.weak['p0']}})
        with self.assertRaisesRegex(ValueError, 'tol'):
            fit_instrument_parameters(self.model, {self.label: self.weak}, tol=0)
