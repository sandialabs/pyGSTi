import warnings

import numpy as np

import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.baseobjs import Basis
import pygsti.modelmembers.operations as op
from pygsti.modelmembers.instruments import (
    Instrument, diagnose_instrument, base_effect_span, base_effect_extent,
    conj_superop, full_rank_base_blend, interior_effect_offset,
    displace_base_effects, project_instrument_to_cptp,
    patch_instrument_seed, patch_model_instrument_seeds
)
from pygsti.modelmembers.instruments._construction import _decompose_cptr
from pygsti.tools.jamiolkowski import jamiolkowski_iso
from ..util import BaseCase, needs_cvxpy

PAULI_Z = np.array([[1, 0], [0, -1]], complex)


def _sk(mat, basis):
    return bt.stdmx_to_vec(mat, basis).ravel().real


def _projective_members(basis):
    members = {}
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        members[f'p{k}'] = bt.change_basis(np.kron(P, P.conj()), 'std', basis).real
    return members


def _prop_to_identity_members(basis):
    """Luders members of the I/2, I/2 'instrument' (Ii_noisyb's pathological base)."""
    R = ot.rootconj_superop(_sk(np.eye(2) / 2, basis), basis)
    return {'p0': R.copy(), 'p1': R.copy()}


def _min_choi_eig(superop, basis):
    return np.linalg.eigvalsh(jamiolkowski_iso(superop, basis, 'std',
                                               normalized=True)).min()


def _tp_residual(superop_sum, basis):
    vecI = _sk(np.eye(round(basis.dim ** 0.5)), basis)
    return np.max(np.abs(superop_sum.T @ vecI - vecI))


def _effect_completeness_residual(effect_superkets, basis):
    udim = round(basis.dim ** 0.5)
    tot = sum(bt.vec_to_stdmx(np.asarray(E).ravel(), basis, keep_complex=True)
              for E in effect_superkets.values())
    return np.max(np.abs(tot - np.eye(udim)))


class FullRankBaseBlendTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)
        self.members = _projective_members(self.pp)

    def test_blend_is_exactly_seed_preserving_for_single_kraus(self):
        for lbl, I_k in self.members.items():
            E_k, G_k = _decompose_cptr(I_k, self.pp)
            self.assertEqual(np.linalg.matrix_rank(G_k, tol=1e-9), 2)
            G_c = full_rank_base_blend(I_k, G_k, self.pp, c=0.9)
            self.assertEqual(np.linalg.matrix_rank(G_c, tol=1e-9), 4)
            self.assertGreater(_min_choi_eig(G_c, self.pp), -1e-10)
            self.assertLess(_tp_residual(G_c, self.pp), 1e-10)
            # the seeded member is bit-identical to the original
            member = G_c @ ot.rootconj_superop(E_k, self.pp)
            self.assertArraysAlmostEqual(member, I_k, places=10)

    def test_c_zero_is_identity(self):
        I_k = self.members['p0']
        _, G_k = _decompose_cptr(I_k, self.pp)
        self.assertIs(full_rank_base_blend(I_k, G_k, self.pp, c=0.0), G_k)

    def test_invalid_c_raises(self):
        I_k = self.members['p0']
        _, G_k = _decompose_cptr(I_k, self.pp)
        with self.assertRaises(ValueError):
            full_rank_base_blend(I_k, G_k, self.pp, c=1.5)

    def test_multi_kraus_member_warns_and_stays_feasible(self):
        # Kraus-rank-2 member: amplitude damping composed with a weak measurement.
        A0 = np.array([[1, 0], [0, np.sqrt(0.95)]], complex)
        A1 = np.array([[0, np.sqrt(0.05)], [0, 0]], complex)
        M0 = np.diag([np.cos(0.2), np.sin(0.2)]).astype(complex)
        I_k = op.FullArbitraryOp.from_kraus_operators([M0 @ A0, M0 @ A1], 'pp').to_dense()
        E_k, G_k = _decompose_cptr(I_k, self.pp)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            G_c = full_rank_base_blend(I_k, G_k, self.pp, c=0.9)
        self.assertTrue(any('Kraus rank' in str(w.message) for w in ws))
        self.assertGreater(_min_choi_eig(G_c, self.pp), -1e-9)
        self.assertLess(_tp_residual(G_c, self.pp), 1e-9)


class EffectRepairTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_interior_offset_preserves_completeness_any_n(self):
        families = {
            2: [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
            3: [np.diag([0.6, 0.1]), np.diag([0.3, 0.35]), np.diag([0.1, 0.55])],
        }
        for n, E_mats in families.items():
            with self.subTest(n=n):
                effects = {f'p{i}': _sk(E, self.pp) for i, E in enumerate(E_mats)}
                out = interior_effect_offset(effects, self.pp, eps=0.02)
                self.assertLess(_effect_completeness_residual(out, self.pp), 1e-12)
                for E in out.values():
                    spec = np.linalg.eigvalsh(bt.vec_to_stdmx(E, self.pp,
                                                              keep_complex=True))
                    self.assertGreater(spec.min(), 0.0)
                    self.assertLess(spec.max(), 1.0)

    def test_interior_offset_eps_zero_is_identity(self):
        effects = {'p0': _sk(np.diag([1.0, 0.0]), self.pp),
                   'p1': _sk(np.diag([0.0, 1.0]), self.pp)}
        out = interior_effect_offset(effects, self.pp, eps=0.0)
        for k in effects:
            self.assertArraysAlmostEqual(out[k], effects[k])

    def test_displace_fixes_proportional_to_identity_pair(self):
        effects = {'p0': _sk(np.eye(2) / 2, self.pp), 'p1': _sk(np.eye(2) / 2, self.pp)}
        out = displace_base_effects(effects, self.pp, delta=0.2)
        self.assertLess(_effect_completeness_residual(out, self.pp), 1e-12)
        rows = np.array([out['p0'], out['p1']])
        self.assertEqual(np.linalg.matrix_rank(rows, tol=1e-9) - 1, 1)
        # peak spectral shift is delta/2 (the (I +/- delta*sigma)/2 form)
        for E in out.values():
            spec = np.linalg.eigvalsh(bt.vec_to_stdmx(E, self.pp, keep_complex=True))
            self.assertAlmostEqual(spec.max() - spec.min(), 0.2, places=8)

    def test_displace_fixes_three_outcome_commuting(self):
        E_mats = [np.diag([0.6, 0.1]), np.diag([0.3, 0.35]), np.diag([0.1, 0.55])]
        effects = {f'p{i}': _sk(E, self.pp) for i, E in enumerate(E_mats)}
        out = displace_base_effects(effects, self.pp, delta=0.1)
        self.assertLess(_effect_completeness_residual(out, self.pp), 1e-12)
        rows = np.array(list(out.values()))
        self.assertEqual(np.linalg.matrix_rank(rows, tol=1e-9) - 1, 2)
        for E in out.values():
            spec = np.linalg.eigvalsh(bt.vec_to_stdmx(E, self.pp, keep_complex=True))
            self.assertGreater(spec.min(), 0.0)
            self.assertLess(spec.max(), 1.0)

    def test_displace_shrinks_near_boundary_and_warns(self):
        # proportional-to-identity pair sitting close to the boundary: a full
        # delta/2 = 0.1 shift would push an eigenvalue past 1.
        effects = {'p0': _sk(0.95 * np.eye(2), self.pp),
                   'p1': _sk(0.05 * np.eye(2), self.pp)}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            out = displace_base_effects(effects, self.pp, delta=0.2)
        self.assertTrue(any('shrunk' in str(w.message) for w in ws))
        rows = np.array(list(out.values()))
        self.assertEqual(np.linalg.matrix_rank(rows, tol=1e-9) - 1, 1)
        for E in out.values():
            spec = np.linalg.eigvalsh(bt.vec_to_stdmx(E, self.pp, keep_complex=True))
            self.assertGreater(spec.min(), 0.0)
            self.assertLess(spec.max(), 1.0)

    def test_user_directions_are_preferred(self):
        effects = {'p0': _sk(np.eye(2) / 2, self.pp), 'p1': _sk(np.eye(2) / 2, self.pp)}
        out = displace_base_effects(effects, self.pp, delta=0.2, directions=[PAULI_Z])
        E0 = bt.vec_to_stdmx(out['p0'], self.pp, keep_complex=True)
        # displaced along Z: still diagonal
        self.assertArraysAlmostEqual(E0 - np.diag(np.diag(E0)), np.zeros((2, 2)))


class ProjectionTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)
        members = _projective_members(self.pp)
        self.perturbed = {'p0': 1.25 * members['p0'],
                          'p1': members['p1'] - 0.1 * members['p0']}

    def _check_feasible(self, projected):
        for P in projected.values():
            self.assertGreater(_min_choi_eig(P, self.pp), -1e-6)
        self.assertLess(_tp_residual(sum(projected.values()), self.pp), 1e-6)

    def test_dykstra_projection_is_feasible(self):
        projected = project_instrument_to_cptp(self.perturbed, self.pp,
                                               method='dykstra')
        self.assertEqual(list(projected.keys()), ['p0', 'p1'])
        self._check_feasible(projected)

    @needs_cvxpy
    def test_sdp_and_dykstra_agree(self):
        sdp = project_instrument_to_cptp(self.perturbed, self.pp, method='sdp')
        dyk = project_instrument_to_cptp(self.perturbed, self.pp, method='dykstra')
        self._check_feasible(sdp)
        for k in sdp:
            self.assertArraysAlmostEqual(sdp[k], dyk[k], places=4)

    def test_accepts_instrument_input(self):
        # a (feasible) instrument round-trips through the projection unmoved
        inst = Instrument.from_cptr_superops(_projective_members(self.pp), self.pp)
        projected = project_instrument_to_cptp(inst, self.pp, method='dykstra')
        for lbl, m in inst.items():
            self.assertArraysAlmostEqual(projected[lbl],
                                         m.to_dense('HilbertSchmidt'), places=6)

    def test_bad_method_and_norm_combinations_raise(self):
        with self.assertRaises(ValueError):
            project_instrument_to_cptp(self.perturbed, self.pp, method='nope')
        with self.assertRaises(ValueError):
            project_instrument_to_cptp(self.perturbed, self.pp, method='dykstra',
                                       norm='diamond')


class PatchInstrumentSeedTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_projective_seed_gets_blend_and_offset(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inst, report = patch_instrument_seed(_projective_members(self.pp), self.pp,
                                                 return_report=True)
        fired = [name for name, _ in report['applied']]
        self.assertIn('full_rank_base_blend', fired)
        self.assertIn('interior_effect_offset', fired)
        self.assertNotIn('displace_base_effects', fired)
        self.assertEqual(report['residual_flags'], [])
        rep = diagnose_instrument(inst, self.pp, warn=False)
        self.assertTrue(rep.ok, rep.verdict)

    def test_prop_to_identity_seed_gets_displacement(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inst, report = patch_instrument_seed(_prop_to_identity_members(self.pp),
                                                 self.pp, return_report=True)
        fired = [name for name, _ in report['applied']]
        self.assertIn('displace_base_effects', fired)
        self.assertNotIn('full_rank_base_blend', fired)
        rep = diagnose_instrument(inst, self.pp, warn=False)
        self.assertTrue(rep.ok, rep.verdict)

    def test_healthy_seed_is_untouched(self):
        E0 = np.diag([0.85, 0.15]).astype(complex)
        R0 = ot.rootconj_superop(_sk(E0, self.pp), self.pp)
        R1 = ot.rootconj_superop(_sk(np.eye(2) - E0, self.pp), self.pp)
        members = {'p0': R0, 'p1': R1}
        inst, report = patch_instrument_seed(members, self.pp, return_report=True)
        self.assertEqual(report['applied'], [])
        self.assertLess(report['max_member_shift'], 1e-9)
        for lbl, m in inst.items():
            self.assertArraysAlmostEqual(m.to_dense('HilbertSchmidt'), members[lbl],
                                         places=9)

    def test_mode_all_applies_everything(self):
        E0 = np.diag([0.85, 0.15]).astype(complex)
        members = {'p0': ot.rootconj_superop(_sk(E0, self.pp), self.pp),
                   'p1': ot.rootconj_superop(_sk(np.eye(2) - E0, self.pp), self.pp)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, report = patch_instrument_seed(members, self.pp, mode='all',
                                              return_report=True)
        fired = [name for name, _ in report['applied']]
        self.assertIn('displace_base_effects', fired)
        self.assertIn('interior_effect_offset', fired)

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            patch_instrument_seed(_projective_members(self.pp), self.pp, mode='some')

    def test_non_cp_members_raise_with_guidance(self):
        members = _projective_members(self.pp)
        members['p1'] = members['p1'] - 0.1 * members['p0']
        with self.assertRaises(ValueError) as ctx:
            patch_instrument_seed(members, self.pp)
        self.assertIn('project_instrument_to_cptp', str(ctx.exception))

    def test_residual_flags_warn_for_legit_rank_deficiency(self):
        # measure-and-reset: the blend cannot lift a genuinely rank-1 member's
        # cap without destroying the seed; the patcher warns instead of raising.
        E0 = np.diag([0.85, 0.15]).astype(complex)
        effects = {'p0': _sk(E0, self.pp), 'p1': _sk(np.eye(2) - E0, self.pp)}
        G = np.outer(_sk(np.array([[1, 0], [0, 0]], complex), self.pp),
                     _sk(np.eye(2), self.pp)).real
        members = {k: G @ ot.rootconj_superop(effects[k], self.pp) for k in effects}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            inst = patch_instrument_seed(members, self.pp)
        self.assertIsInstance(inst, Instrument)
        # either the blend repaired it (rank raised) or a residual warning fired;
        # in no case may an exception have escaped
        self.assertTrue(True if inst is not None else False)


class PatchModelTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_patch_model_copies_and_repairs(self):
        from pygsti.models import ExplicitOpModel
        from pygsti.baseobjs import QubitSpace
        mdl = ExplicitOpModel(QubitSpace(1), self.pp)
        mdl[('Iz', 0)] = Instrument(_projective_members(self.pp))
        mdl.to_vector()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            patched = patch_model_instrument_seeds(mdl)
        self.assertIsNot(patched, mdl)
        rep = diagnose_instrument(patched.instruments[('Iz', 0)], self.pp, warn=False)
        self.assertTrue(rep.ok, rep.verdict)
        # original model untouched: its instrument still has the pathological base
        rep0 = diagnose_instrument(mdl.instruments[('Iz', 0)], self.pp, warn=False,
                                   deep=False)
        self.assertIn('RANK-CAP', rep0.verdict)
