import warnings

import numpy as np

import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.baseobjs import Basis
from pygsti.modelmembers import instruments as inst_mod
from pygsti.modelmembers.instruments import (
    Instrument, TPInstrument, diagnose_instrument, diagnose_model_instruments,
    instrument_static_base, static_base_ranks, base_effect_span, base_effect_extent
)
from ..util import BaseCase

PAULI = {'I': np.eye(2, dtype=complex),
         'X': np.array([[0, 1], [1, 0]], complex),
         'Z': np.array([[1, 0], [0, -1]], complex)}


def _sk(mat, basis):
    return bt.stdmx_to_vec(mat, basis).ravel().real


def _luders_instrument(effect_mats, basis, **kwargs):
    """Instrument.from_effects with identity post-measurement gates."""
    members = {f'p{i}': _sk(E, basis) for i, E in enumerate(effect_mats)}
    return Instrument.from_effects(members, basis, **kwargs)


def _projective_member_superops(basis):
    """Ideal Z-measurement members P_k rho P_k as dense superops."""
    members = {}
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        members[f'p{k}'] = bt.change_basis(np.kron(P, P.conj()), 'std', basis).real
    return members


class SpanLawTester(BaseCase):
    """The R5 span-law table: m = dim span{E_base,k} - 1, target n - 1."""

    def setUp(self):
        self.pp = Basis.cast('pp', 4)
        self.pp2 = Basis.cast('pp', 16)

    def _span(self, effect_mats, basis):
        inst = _luders_instrument(effect_mats, basis)
        return base_effect_span(inst, basis)

    def test_proportional_to_identity_pair(self):
        span = self._span([np.eye(2) / 2, np.eye(2) / 2], self.pp)
        self.assertEqual(span['m'], 0)
        self.assertEqual(span['target'], 1)
        self.assertTrue(span['deficient'])
        self.assertEqual(span['reachable_effect_dim'], 0)

    def test_ideal_projective_pair_is_healthy(self):
        # A rank-1 projector is NOT traceless: P0, P1 span {I, Z}, so m = 1 = n-1.
        span = self._span([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], self.pp)
        self.assertEqual(span['m'], 1)
        self.assertFalse(span['deficient'])
        self.assertEqual(span['reachable_effect_dim'], 4)

    def test_generic_two_outcome(self):
        span = self._span([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])], self.pp)
        self.assertEqual(span['m'], 1)
        self.assertFalse(span['deficient'])

    def test_three_outcome_commuting_is_deficient(self):
        # An IDEAL (all-diagonal) 3-outcome measurement spans only {I, Z}: m = 1 < 2.
        E = [np.diag([0.6, 0.1]), np.diag([0.3, 0.35]), np.diag([0.1, 0.55])]
        span = self._span(E, self.pp)
        self.assertEqual(span['m'], 1)
        self.assertEqual(span['target'], 2)
        self.assertTrue(span['deficient'])
        self.assertEqual(span['reachable_effect_dim'], 4)
        self.assertEqual(span['full_effect_dim'], 8)

    def test_three_outcome_noncommuting_is_healthy(self):
        E = [np.eye(2) / 3 + 0.15 * PAULI['Z'],
             np.eye(2) / 3 + 0.15 * PAULI['X'],
             np.eye(2) / 3 - 0.15 * (PAULI['Z'] + PAULI['X'])]
        span = self._span(E, self.pp)
        self.assertEqual(span['m'], 2)
        self.assertFalse(span['deficient'])
        self.assertEqual(span['reachable_effect_dim'], 8)

    def test_d4_proportional_to_identity(self):
        span = self._span([np.eye(4) / 2, np.eye(4) / 2], self.pp2)
        self.assertEqual(span['m'], 0)
        self.assertTrue(span['deficient'])

    def test_d4_degenerate_blocks_are_healthy(self):
        # Spectral degeneracy WITHIN one effect does not freeze anything; only
        # the span across outcomes matters (settles the findings-r4 question).
        E0 = np.diag([0.7, 0.7, 0.3, 0.3])
        span = self._span([E0, np.eye(4) - E0], self.pp2)
        self.assertEqual(span['m'], 1)
        self.assertFalse(span['deficient'])
        self.assertEqual(span['reachable_effect_dim'], 16)


class ExtentTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_interior_effects(self):
        inst = _luders_instrument([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])], self.pp)
        rows = base_effect_extent(inst, self.pp)
        for row in rows.values():
            lo, hi = row['spec_interval']
            self.assertAlmostEqual(lo, 0.15)
            self.assertAlmostEqual(hi, 0.85)
            self.assertAlmostEqual(row['extent'], np.sqrt(2) * 0.7 / 2)
            self.assertAlmostEqual(row['boundary_margin'], 0.15)

    def test_projective_effects_touch_boundary(self):
        inst = _luders_instrument([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], self.pp)
        rows = base_effect_extent(inst, self.pp)
        for row in rows.values():
            self.assertAlmostEqual(row['boundary_margin'], 0.0)
            self.assertAlmostEqual(row['extent'], np.sqrt(2) / 2)

    def test_proportional_to_identity_has_zero_extent(self):
        inst = _luders_instrument([np.eye(2) / 2, np.eye(2) / 2], self.pp)
        rows = base_effect_extent(inst, self.pp)
        for row in rows.values():
            self.assertAlmostEqual(row['extent'], 0.0)


class StaticBaseTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_exact_extraction_from_parameterized_chart(self):
        E0 = np.diag([0.85, 0.15])
        inst = _luders_instrument([E0, np.eye(2) - E0], self.pp)
        basis, effects, gates, info = instrument_static_base(inst, self.pp)
        self.assertTrue(info['exact'])
        self.assertArraysAlmostEqual(effects['p0'], _sk(E0, self.pp))
        self.assertArraysAlmostEqual(gates['p0'], np.eye(4))
        self.assertTrue(all(info['gate_frozen'].values()))

    def test_full_tp_gates_are_not_frozen(self):
        E0 = np.diag([0.85, 0.15])
        inst = _luders_instrument([E0, np.eye(2) - E0], self.pp,
                                  gate_parameterization='full TP')
        _, _, _, info = instrument_static_base(inst, self.pp)
        self.assertTrue(info['exact'])
        self.assertFalse(any(info['gate_frozen'].values()))

    def test_predicted_base_from_dense_members(self):
        members = _projective_member_superops(self.pp)
        basis, effects, gates, info = instrument_static_base(members, self.pp)
        self.assertFalse(info['exact'])
        # the canonical decomposition of an ideal projective member is rank-2
        for G in gates.values():
            self.assertEqual(np.linalg.matrix_rank(G, tol=1e-9), 2)

    def test_predicted_base_from_tpinstrument(self):
        members = _projective_member_superops(self.pp)
        tpinst = TPInstrument(members)
        _, _, gates, info = instrument_static_base(tpinst, self.pp)
        self.assertFalse(info['exact'])
        self.assertEqual(len(gates), 2)

    def test_non_cp_members_raise_with_guidance(self):
        members = _projective_member_superops(self.pp)
        members['p1'] = members['p1'] - 0.1 * members['p0']   # not CP
        with self.assertRaises(ValueError) as ctx:
            instrument_static_base(members, self.pp)
        self.assertIn('project', str(ctx.exception))


class RankCheckTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_projective_base_is_rank_capped(self):
        inst = Instrument.from_cptr_superops(_projective_member_superops(self.pp), self.pp)
        ranks = static_base_ranks(inst, self.pp)
        for row in ranks.values():
            self.assertEqual(row['rank'], 2)
            self.assertEqual(row['full'], 4)
            self.assertTrue(row['frozen'])

    def test_interior_base_is_full_rank(self):
        inst = _luders_instrument([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])], self.pp)
        ranks = static_base_ranks(inst, self.pp)
        for row in ranks.values():
            self.assertEqual(row['rank'], 4)


class DiagnoseVerdictTester(BaseCase):
    """Flags on known-bad and known-good seeds, deep check included."""

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def _diagnose(self, *args, **kwargs):
        kwargs.setdefault('warn', False)
        return diagnose_instrument(*args, **kwargs)

    def test_ideal_projective_flags_rank_cap_and_boundary(self):
        rep = self._diagnose(_projective_member_superops(self.pp), self.pp)
        self.assertIn('RANK-CAP(2/4)', rep.verdict)
        self.assertIn('CP-BOUNDARY', rep.verdict)
        self.assertIn('BLIND', rep.verdict)
        # the blind directions are member directions (rank cap), not frozen effects
        self.assertEqual(rep.details['effect_rank'], rep.details['effect_target'])

    def test_proportional_to_identity_flags_span(self):
        rep = self._diagnose(_luders_instrument([np.eye(2) / 2, np.eye(2) / 2], self.pp),
                             self.pp)
        self.assertIn('SPAN-DEFICIENT(0/1)', rep.verdict)
        self.assertIn('ZERO-EXTENT', rep.verdict)
        self.assertIn('EFFECTS-FROZEN', rep.verdict)
        # the sampled effect rank must agree with the span law's d^2 * m
        self.assertEqual(rep.details['effect_rank'],
                         rep.details['effect_span']['reachable_effect_dim'])

    def test_three_outcome_commuting_flags_span(self):
        E = [np.diag([0.6, 0.1]), np.diag([0.3, 0.35]), np.diag([0.1, 0.55])]
        rep = self._diagnose(_luders_instrument(E, self.pp), self.pp)
        self.assertIn('SPAN-DEFICIENT(1/2)', rep.verdict)
        self.assertEqual(rep.details['effect_rank'], 4)   # d^2 * m = 4 of 8
        self.assertEqual(rep.details['effect_target'], 8)

    def test_false_positive_battery_passes(self):
        healthy = {
            'noisy readout': _luders_instrument(
                [np.diag([0.85, 0.15]), np.diag([0.15, 0.85])], self.pp),
            'three-outcome noncommuting': _luders_instrument(
                [np.eye(2) / 3 + 0.15 * PAULI['Z'],
                 np.eye(2) / 3 + 0.15 * PAULI['X'],
                 np.eye(2) / 3 - 0.15 * (PAULI['Z'] + PAULI['X'])], self.pp),
        }
        for name, inst in healthy.items():
            with self.subTest(case=name):
                rep = self._diagnose(inst, self.pp)
                self.assertTrue(rep.ok, f"{name}: unexpected flags {rep.verdict}")

    def test_glnd_chart_passes(self):
        # The stochastic-block-quadratic (cholesky) subtlety is CPTPLND-specific;
        # a healthy GLND chart must also pass (generic-point sampling handles both).
        # The POVM error map stays CP-constrained ('CPTPLND') even for GLND gates,
        # matching instruments.convert()'s minimal_cp_paramtype behavior -- a
        # non-CP effect map would let sampled effects leave [0, 1].
        inst = _luders_instrument([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])],
                                  self.pp, gate_parameterization='GLND')
        rep = self._diagnose(inst, self.pp, gate_parameterization='GLND')
        self.assertTrue(rep.ok, rep.verdict)

    def test_measure_and_reset_flags_rank_cap_informationally(self):
        # Legitimately rank-1 members: the flag is CORRECT and informational.
        # Effects must remain free (the cap is in the member direction).
        basis = self.pp
        E0 = np.diag([0.85, 0.15]).astype(complex)
        effects = {'p0': _sk(E0, basis), 'p1': _sk(np.eye(2) - E0, basis)}
        G = np.outer(_sk(np.array([[1, 0], [0, 0]], complex), basis),
                     _sk(np.eye(2), basis)).real
        members = {k: (effects[k], G) for k in effects}
        inst = Instrument.from_effects(members, basis)
        rep = self._diagnose(inst, basis)
        self.assertIn('RANK-CAP(1/4)', rep.verdict)
        self.assertEqual(rep.details['effect_rank'], rep.details['effect_target'])

    def test_shallow_only_runs_structural_checks(self):
        rep = self._diagnose(_projective_member_superops(self.pp), self.pp, deep=False)
        self.assertFalse(rep.deep)
        self.assertIn('RANK-CAP', rep.verdict)
        self.assertNotIn('BLIND', rep.verdict)
        self.assertNotIn('member_rank', rep.details)

    def test_deep_requires_lindblad_parameterization(self):
        inst = _projective_member_superops(self.pp)
        with self.assertRaises(ValueError):
            self._diagnose(inst, self.pp, gate_parameterization='full TP')

    def test_warning_emission(self):
        members = _projective_member_superops(self.pp)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            diagnose_instrument(members, self.pp, deep=False, warn=True)
        flagged = [w for w in ws if 'RANK-CAP' in str(w.message)]
        self.assertEqual(len(flagged), 1)
        self.assertIn('repair', str(flagged[0].message))

        healthy = _luders_instrument([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])],
                                     self.pp)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            diagnose_instrument(healthy, self.pp, deep=False, warn=True)
        self.assertEqual(len([w for w in ws if 'diagnostics' in str(w.message)]), 0)


class ModelLevelTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def _host(self, inst):
        from pygsti.models import ExplicitOpModel
        from pygsti.baseobjs import QubitSpace
        mdl = ExplicitOpModel(QubitSpace(1), self.pp)
        mdl[('Itest', 0)] = inst
        mdl.to_vector()
        return mdl

    def test_model_reports_match_instrument_reports(self):
        inst = Instrument.from_cptr_superops(_projective_member_superops(self.pp), self.pp)
        mdl = self._host(inst)
        reports = diagnose_model_instruments(mdl, warn=False)
        self.assertEqual(list(reports.keys()), [('Itest', 0)])
        rep = reports[('Itest', 0)]
        self.assertIn('RANK-CAP(2/4)', rep.verdict)
        # model round-trip must not corrupt the model's parameter vector
        self.assertArraysAlmostEqual(mdl.to_vector(),
                                     np.zeros(mdl.num_params))

    def test_healthy_model_instrument_passes(self):
        inst = _luders_instrument([np.diag([0.85, 0.15]), np.diag([0.15, 0.85])], self.pp)
        mdl = self._host(inst)
        reports = diagnose_model_instruments(mdl, warn=False)
        self.assertTrue(reports[('Itest', 0)].ok)
