import warnings

import numpy as np

import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.baseobjs import Basis
import pygsti.modelmembers.operations as op
from pygsti.modelmembers.instruments import (
    Instrument, diagnose_instrument, displace_base_effects, project_instrument_to_cptp,
    patch_instrument_seed, patch_model_instrument_seeds
)
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


class EffectRepairTester(BaseCase):

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

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

    def test_projective_seed_gets_identity_unitaries_and_keeps_its_effects(self):
        inst, report = patch_instrument_seed(_projective_members(self.pp), self.pp,
                                             return_report=True)
        fired = [name for name, _ in report['applied']]
        self.assertEqual(fired, ['infer_unitary_factors'])
        self.assertEqual(report['residual_flags'], [])
        self.assertLess(report['max_member_shift'], 1e-12)
        for member in inst.values():
            static = member.factorops[1].factorops[0]
            self.assertIsInstance(static, op.StaticUnitaryOp)
            self.assertArraysAlmostEqual(static.to_dense('HilbertSchmidt'), np.eye(4))
        # The projective effects stay put, so the ideal measurement stays reachable; the only
        # remaining flag concerns the starting point, which prepare_lindblad_seed handles.
        rep = diagnose_instrument(inst, self.pp, warn=False)
        self.assertEqual([f.name for f in rep.flags], ['CP-BOUNDARY'])
        self.assertIn('prepare_lindblad_seed', rep.flags[0].repair)

    def test_prop_to_identity_seed_gets_displacement(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inst, report = patch_instrument_seed(_prop_to_identity_members(self.pp),
                                                 self.pp, return_report=True)
        fired = [name for name, _ in report['applied']]
        self.assertIn('displace_base_effects', fired)
        rep = diagnose_instrument(inst, self.pp, warn=False)
        self.assertTrue(rep.ok, rep.verdict)

    def test_healthy_seed_is_untouched(self):
        E0 = np.diag([0.85, 0.15]).astype(complex)
        R0 = ot.rootconj_superop(_sk(E0, self.pp), self.pp)
        R1 = ot.rootconj_superop(_sk(np.eye(2) - E0, self.pp), self.pp)
        members = {'p0': R0, 'p1': R1}
        inst, report = patch_instrument_seed(members, self.pp, return_report=True)
        self.assertEqual(report['applied'], [('infer_unitary_factors', None)])
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

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            patch_instrument_seed(_projective_members(self.pp), self.pp, mode='some')

    def test_non_cp_members_raise_with_guidance(self):
        members = _projective_members(self.pp)
        members['p1'] = members['p1'] - 0.1 * members['p0']
        with self.assertRaises(ValueError) as ctx:
            patch_instrument_seed(members, self.pp)
        self.assertIn('project_instrument_to_cptp', str(ctx.exception))

    def test_multi_kraus_members_are_approximated_with_their_effects_kept(self):
        # Weak measurement followed by a reset to |0>: each member has two Kraus operators,
        # so the seed is the effect-preserving one-Kraus approximation, not an error.
        E0 = np.diag([0.85, 0.15]).astype(complex)
        effects = {'p0': _sk(E0, self.pp), 'p1': _sk(np.eye(2) - E0, self.pp)}
        G = np.outer(_sk(np.array([[1, 0], [0, 0]], complex), self.pp),
                     _sk(np.eye(2), self.pp)).real
        members = {k: G @ ot.rootconj_superop(effects[k], self.pp) for k in effects}
        inst, report = patch_instrument_seed(members, self.pp, return_report=True)
        for lbl, m in inst.items():
            self.assertGreater(report['unitary_factors'][lbl].residual, 1e-2)
            self.assertArraysAlmostEqual(m.to_dense('HilbertSchmidt')[0], members[lbl][0])


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
        self.assertEqual([f.name for f in rep.flags], ['CP-BOUNDARY'])  # left for prepare_lindblad_seed
        # original model untouched: its instrument is still the plain dense one
        self.assertNotIsInstance(mdl.instruments[('Iz', 0)]['p0'], op.ComposedOp)
