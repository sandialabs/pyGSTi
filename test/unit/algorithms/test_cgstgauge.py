import warnings

import numpy as np
from scipy.linalg import expm

from pygsti.algorithms import cgstgauge
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel
from pygsti.baseobjs.label import Label
from pygsti.models.gaugegroup import FullGaugeGroupElement
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.tools.optools import unitary_to_pauligate as _u2p
from ..util import BaseCase

_sx = np.array([[0, 1], [1, 0]], complex)
_sy = np.array([[0, -1j], [1j, 0]], complex)
_sz = np.array([[1, 0], [0, -1]], complex)


def _rot(axis, angle):
    """PTM of a rotation by `angle` about a Pauli axis."""
    return _u2p(expm(-0.5j * angle * axis))


def _pspec():
    return QubitProcessorSpec(1, ['Gzpi2', 'Gypi2', 'Gi'], qubit_labels=['Q0'])


def _target():
    return create_explicit_model(_pspec(), simulator='matrix')


def _manuscript_model(theta=0., alpha=0., beta=0., lam1=1., lam2=1., a=0.,
                      r1=0., r2=0., cxy=0., cxz=0., cyz=0., ay=0., arel=0.,
                      idle_angles=(0., 0., 0.)):
    """
    The manuscript's {S, sqrt(Y)} model *in* the standard gauge: eq:S_Channel for S and
    the corrected eq:Y_Channel for sqrt(Y) (a_rel and r_2 on the X and Z rows, a_y and
    r_1 on the Y row; the second-order residuals delta, delta' set to zero).
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


_INJECTED = dict(theta=0.02, alpha=-0.015, beta=0.01, lam1=0.99, lam2=0.985, a=0.004,
                 r1=0.012, r2=0.008, cxy=0.002, cxz=-0.001, cyz=0.0015, ay=0.003, arel=0.002,
                 idle_angles=(0.002, 0.003, 0.004))


def _lindblad_model(scale=1.0):
    noise = {
        'Gzpi2:Q0': {('H', 'Z'): 0.01, ('H', 'X'): 0.008, ('S', 'X'): 0.012, ('S', 'Y'): 0.004,
                     ('S', 'Z'): 0.006, ('C', 'X', 'Z'): 0.002, ('A', 'X', 'Z'): 0.003,
                     ('A', 'Y', 'Z'): -0.002},
        'Gypi2:Q0': {('H', 'Y'): 0.004, ('H', 'X'): 0.003, ('S', 'X'): 0.006, ('S', 'Y'): 0.003,
                     ('S', 'Z'): 0.005, ('A', 'X', 'Y'): 0.002},
        'Gi:Q0': {('H', 'Z'): 0.004, ('S', 'X'): 0.002},
    }
    noise = {g: {k: scale * v for k, v in d.items()} for g, d in noise.items()}
    return create_explicit_model(_pspec(), lindblad_error_coeffs=noise,
                                 lindblad_parameterization='GLND', simulator='matrix')


def _gauge_transformed(model, B):
    """Copy of `model` (as 'full TP') with gates B Lam B^-1, preps B rho, effects E B^-1."""
    mdl = model.copy()
    mdl.set_all_parameterizations('full TP')
    mdl.transform_inplace(FullGaugeGroupElement(np.linalg.inv(B)))
    return mdl


def _random_tp_gauge(rng, size):
    K = size * rng.standard_normal((4, 4))
    K[0, :] = 0.0
    return expm(K)


def _max_gate_diff(m1, m2):
    return max(np.abs(m1.operations[lbl].to_dense() - m2.operations[lbl].to_dense()).max()
               for lbl in m1.operations.keys())


class CommutingGaugeTester(BaseCase):

    def setUp(self):
        self.target = _target()
        self.S = self.target.operations[('Gzpi2', 'Q0')].to_dense()

    def test_lindblad_noisy_s_gate(self):
        Lam = _lindblad_model().operations[('Gzpi2', 'Q0')].to_dense()
        A = cgstgauge.commuting_gauge_transform(Lam, self.S)
        self.assertTrue(np.isrealobj(A))
        self.assertArraysAlmostEqual(A[0, :], [1, 0, 0, 0])
        T = A @ Lam @ np.linalg.inv(A)
        self.assertLess(np.abs(T @ self.S - self.S @ T).max(), 1e-10)
        self.assertArraysAlmostEqual(T[0, :], [1, 0, 0, 0])
        ev_T = np.sort_complex(np.round(np.linalg.eigvals(T), 10))
        ev_L = np.sort_complex(np.round(np.linalg.eigvals(Lam), 10))
        self.assertArraysAlmostEqual(ev_T, ev_L)
        # block diagonal w.r.t. the ideal eigenspaces {I, Z} and {X, Y}
        self.assertLess(np.abs(T[np.ix_([0, 3], [1, 2])]).max(), 1e-12)
        self.assertLess(np.abs(T[np.ix_([1, 2], [0, 3])]).max(), 1e-12)

    def test_tends_to_identity_with_vanishing_noise(self):
        # the canonical normalization A = sum_c P_c Pi_c makes A -> 1 as the noise -> 0
        prev = np.inf
        for scale in (1.0, 0.1, 0.01):
            Lam = _lindblad_model(scale).operations[('Gzpi2', 'Q0')].to_dense()
            A = cgstgauge.commuting_gauge_transform(Lam, self.S)
            dist = np.abs(A - np.identity(4)).max()
            self.assertLess(dist, 2 * scale * 0.05)
            self.assertLess(dist, prev)
            prev = dist
        A = cgstgauge.commuting_gauge_transform(self.S, self.S)
        self.assertArraysAlmostEqual(A, np.identity(4))

    def test_already_commuting_model_is_fixed_point(self):
        Lam = _manuscript_model(**_INJECTED).operations[('Gzpi2', 'Q0')].to_dense()
        A = cgstgauge.commuting_gauge_transform(Lam, self.S)
        self.assertArraysAlmostEqual(A, np.identity(4))

    def test_large_noise_raises(self):
        # an over-rotation of 1 rad moves the equatorial eigenvalues beyond half the ideal gap
        Lam = _rot(_sz, np.pi / 2 + 1.0)
        with self.assertRaises(ValueError):
            cgstgauge.commuting_gauge_transform(Lam, self.S)
        # non-TP input is rejected
        with self.assertRaises(ValueError):
            cgstgauge.commuting_gauge_transform(0.9 * self.S, self.S)

    def test_two_qubit_smoke(self):
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gcnot'], geometry='line')
        target = create_explicit_model(pspec, simulator='matrix')
        lbl = [k for k in target.operations.keys() if k.name == 'Gcnot'][0]
        G = target.operations[lbl].to_dense()
        # depolarizing noise times a small coherent two-qubit error, applied after the ideal gate
        U_err = expm(-0.5j * (0.01 * np.kron(_sz, _sx) + 0.005 * np.kron(np.identity(2), _sy)))
        Lam = np.diag([1.0] + [0.98] * 15) @ _u2p(U_err) @ G
        A = cgstgauge.commuting_gauge_transform(Lam, G)
        T = A @ Lam @ np.linalg.inv(A)
        self.assertLess(np.abs(T @ G - G @ T).max(), 1e-9)
        self.assertArraysAlmostEqual(T[0, :], np.identity(16)[0])
        Ks = cgstgauge.tp_commutant_basis(G)
        self.assertGreater(len(Ks), 0)
        for K in Ks:
            self.assertLess(np.abs(K @ G - G @ K).max(), 1e-10)
            self.assertLess(np.abs(K[0, :]).max(), 1e-12)


class TPCommutantBasisTester(BaseCase):

    def test_s_gate_commutant(self):
        S = _target().operations[('Gzpi2', 'Q0')].to_dense()
        Ks = cgstgauge.tp_commutant_basis(S)
        self.assertEqual(len(Ks), 4)
        gram = np.array([[np.sum(K1 * K2) for K2 in Ks] for K1 in Ks])
        self.assertArraysAlmostEqual(gram, np.identity(4))
        for K in Ks:
            self.assertTrue(np.isrealobj(K))
            self.assertLess(np.abs(K @ S - S @ K).max(), 1e-12)
            self.assertLess(np.abs(K[0, :]).max(), 1e-12)
            # exp(K) is a TP gauge transformation commuting with S
            E = expm(0.3 * K)
            self.assertArraysAlmostEqual(E[0, :], [1, 0, 0, 0])
            self.assertLess(np.abs(E @ S - S @ E).max(), 1e-12)
        # the manuscript's A_1, A_2, A_3 generators all lie in the span
        span = np.column_stack([K.ravel() for K in Ks])
        proj = span @ span.T
        gens = [np.diag([0, 1, 1, 0]),                         # xi : equatorial scale
                np.diag([0, 0, 0, 1]),                         # zeta : axis scale
                np.outer([0, 0, 0, 1], [1, 0, 0, 0]),          # eta : axis shift
                np.outer([0, 0, 1, 0], [0, 1, 0, 0]) - np.outer([0, 1, 0, 0], [0, 0, 1, 0])]  # psi
        for g in gens:
            self.assertArraysAlmostEqual(proj @ g.ravel(), g.ravel())

    def test_pi_rotation_has_larger_commutant(self):
        Ks = cgstgauge.tp_commutant_basis(_rot(_sy, np.pi))
        self.assertEqual(len(Ks), 6)


class StandardGaugeTester(BaseCase):

    def setUp(self):
        self.target = _target()
        self.model = _manuscript_model(**_INJECTED)
        self.rng = np.random.default_rng(2026)

    def test_manuscript_model_is_nearly_fixed(self):
        # the corrected manuscript form is a first-order stationary point of the stage-2
        # objective: the transformation is within O(eps^2) of the identity and only the
        # second-order residuals delta, delta' of the sqrt(Y) channel change
        A, info = cgstgauge.standard_gauge_transform(self.model, self.target, 'Gzpi2', ['Gypi2'],
                                                     return_info=True)
        self.assertArraysAlmostEqual(info['stage1'], np.identity(4))
        self.assertLess(np.abs(A - np.identity(4)).max(), 1e-4)
        self.assertEqual(len(info['basis']), 4)
        self.assertEqual(len(info['gate_fixed_basis']), 3)
        self.assertEqual(len(info['spam_fixed_basis']), 1)
        self.assertEqual(info['reference_gate'], Label('Gzpi2', 'Q0'))
        self.assertEqual(info['other_gates'], [Label('Gypi2', 'Q0')])
        # the SPAM-only direction is the uniform Bloch scaling diag(0,1,1,1)
        D = info['spam_fixed_basis'][0]
        self.assertArraysAlmostEqual(np.abs(D) * np.sqrt(3), np.diag([0, 1, 1, 1]))
        fixed = cgstgauge.fix_standard_gauge(self.model, self.target, 'Gzpi2', ['Gypi2'])
        self.assertLess(_max_gate_diff(fixed, self.model), 1e-4)
        Y = fixed.operations[('Gypi2', 'Q0')].to_dense()
        # symmetric structure of the corrected eq:Y_Channel: equal X/Z affine entries and decays
        self.assertLess(abs(Y[1, 0] - Y[3, 0]), 1e-4)
        self.assertLess(abs(Y[1, 3] + Y[3, 1]), 1e-4)  # E_XX == E_ZZ  (Lam = E_sto Y)

    def test_gauge_invariance_and_recovery(self):
        fixed0 = cgstgauge.fix_standard_gauge(self.model, self.target, 'Gzpi2', ['Gypi2'])
        for size in (1e-2, 3e-2):
            B = _random_tp_gauge(self.rng, size)
            moved = _gauge_transformed(self.model, B)
            self.assertGreater(_max_gate_diff(moved, self.model), 1e-3)  # actually moved
            fixed = cgstgauge.fix_standard_gauge(moved, self.target, 'Gzpi2', ['Gypi2'])
            # exact gauge invariance of the fixing (gates and SPAM)
            self.assertLess(_max_gate_diff(fixed, fixed0), 1e-8)
            self.assertArraysAlmostEqual(fixed.preps['rho0'].to_dense(), fixed0.preps['rho0'].to_dense())
            for elbl in fixed.povms['Mdefault'].keys():
                self.assertArraysAlmostEqual(fixed.povms['Mdefault'][elbl].to_dense(),
                                             fixed0.povms['Mdefault'][elbl].to_dense())
            # recovery of the manuscript matrices up to their second-order residuals
            self.assertLess(_max_gate_diff(fixed, self.model), 1e-4)
            # the S gate is exactly in the commuting form again and the model is TP
            S = self.target.operations[('Gzpi2', 'Q0')].to_dense()
            LamS = fixed.operations[('Gzpi2', 'Q0')].to_dense()
            self.assertLess(np.abs(LamS @ S - S @ LamS).max(), 1e-10)
            for lbl in fixed.operations.keys():
                self.assertArraysAlmostEqual(fixed.operations[lbl].to_dense()[0, :], [1, 0, 0, 0])
        # idempotency
        again = cgstgauge.fix_standard_gauge(fixed0, self.target, 'Gzpi2', ['Gypi2'])
        self.assertLess(_max_gate_diff(again, fixed0), 1e-8)

    def test_psi_convention_tilt_in_yz_plane(self):
        # A pure Y-Z tilt X(beta) Y(pi/2) X(-beta) has an error generator with equal
        # H_X and H_Z components.  Tilting the sqrt(Y) axis about the equatorial axis
        # (cos phi, 0, sin phi) tilts it toward Z by beta*cos(phi) (relational, kept) and
        # toward X by beta*sin(phi) (pure gauge relative to S, removed by the equatorial
        # rotation psi).  After fixing, the tilt must lie in the Y-Z plane with size
        # proportional to cos(phi), while the S gate defines the Z axis.
        beta = 0.02
        hx_ref = None
        for phi in (0.0, 0.7, 2.0, -1.3, np.pi / 2):
            axis = np.cos(phi) * _sx + np.sin(phi) * _sz   # tilt about an equatorial axis
            model = _manuscript_model(theta=0.01, lam1=0.995, lam2=0.99, r1=0.006, r2=0.004)
            model.operations[('Gypi2', 'Q0')] = _rot(axis, beta) @ model.operations[('Gypi2', 'Q0')].to_dense() @ _rot(axis, -beta)
            fixed = cgstgauge.fix_standard_gauge(model, self.target, 'Gzpi2', ['Gypi2'])
            co = cgstgauge.errorgen_coefficients_in_gauge(fixed, self.target)[Label('Gypi2', 'Q0')]
            hx = co[GlobalElementaryErrorgenLabel('H', ('X',), ('Q0',))]
            hz = co[GlobalElementaryErrorgenLabel('H', ('Z',), ('Q0',))]
            self.assertLess(abs(hx - hz), 1e-4, msg="phi=%g: H_X=%g H_Z=%g" % (phi, hx, hz))
            if hx_ref is None:  # phi = 0: the pure Y-Z tilt is kept in full
                hx_ref = hx
                self.assertGreater(abs(hx), beta / 10)
            else:
                self.assertLess(abs(hx - hx_ref * np.cos(phi)), 1e-4,
                                msg="phi=%g: H_X=%g ref=%g" % (phi, hx, hx_ref))
            # the S gate carries no X/Y Hamiltonian in the standard gauge
            coS = cgstgauge.errorgen_coefficients_in_gauge(fixed, self.target)[Label('Gzpi2', 'Q0')]
            self.assertLess(abs(coS[GlobalElementaryErrorgenLabel('H', ('X',), ('Q0',))]), 1e-10)
            self.assertLess(abs(coS[GlobalElementaryErrorgenLabel('H', ('Y',), ('Q0',))]), 1e-10)

    def test_lindblad_model_and_label_resolution(self):
        model = _lindblad_model()
        A1 = cgstgauge.standard_gauge_transform(model, self.target, 'Gzpi2')
        A2 = cgstgauge.standard_gauge_transform(model, self.target, Label('Gzpi2', 'Q0'),
                                                [('Gypi2', 'Q0'), Label('Gi', 'Q0')])
        A3 = cgstgauge.standard_gauge_transform(model, self.target, ('Gzpi2', 'Q0'), ['Gypi2', 'Gi'])
        self.assertArraysAlmostEqual(A1, A2)
        self.assertArraysAlmostEqual(A1, A3)
        with self.assertRaises(KeyError):
            cgstgauge.standard_gauge_transform(model, self.target, 'Gfoo')
        fixed = cgstgauge.fix_standard_gauge(model, self.target, 'Gzpi2')
        S = self.target.operations[('Gzpi2', 'Q0')].to_dense()
        LamS = fixed.operations[('Gzpi2', 'Q0')].to_dense()
        self.assertLess(np.abs(LamS @ S - S @ LamS).max(), 1e-10)
        # gauge invariance with default other_gates (includes the idle)
        B = _random_tp_gauge(self.rng, 1e-2)
        fixed_moved = cgstgauge.fix_standard_gauge(_gauge_transformed(model, B), self.target, 'Gzpi2')
        self.assertLess(_max_gate_diff(fixed_moved, fixed), 1e-8)
        # the S gate's coefficients in the commuting gauge: only H_Z, S_X = S_Y, S_Z and the
        # axis active error survive
        co = cgstgauge.errorgen_coefficients_in_gauge(fixed, self.target)[Label('Gzpi2', 'Q0')]
        g = lambda t, *bels: co[GlobalElementaryErrorgenLabel(t, bels, ('Q0',))]
        self.assertLess(abs(g('H', 'X')), 1e-10)
        self.assertLess(abs(g('H', 'Y')), 1e-10)
        self.assertLess(abs(g('S', 'X') - g('S', 'Y')), 1e-10)
        self.assertLess(abs(g('S', 'X') - 0.008), 1e-4)  # mean of the injected 0.012 and 0.004
        self.assertLess(abs(g('H', 'Z') - 0.01), 1e-4)
        for bels in (('X', 'Y'), ('X', 'Z'), ('Y', 'Z')):
            self.assertLess(abs(g('C', *bels)), 1e-10)
        self.assertLess(abs(g('A', 'X', 'Z')), 1e-10)
        self.assertLess(abs(g('A', 'Y', 'Z')), 1e-10)

    def test_well_posed_gate_set_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cgstgauge.standard_gauge_transform(_lindblad_model(), self.target, 'Gzpi2')
        self.assertEqual([str(w.message) for w in caught], [])

    def test_reducible_gate_set_warns(self):
        # {Z(pi), Y(pi/2)} leaves the Y axis invariant, so the Y-axis scaling commutes
        # with every ideal gate and -- unlike the uniform Bloch scaling -- is invisible
        # to the ideal SPAM: two SPAM-only directions, and the standard gauge is not
        # gauge invariant.  That must be flagged.
        pspec = QubitProcessorSpec(1, ['Gzpi', 'Gypi2'], qubit_labels=['Q0'])
        target = create_explicit_model(pspec, simulator='matrix')
        noise = {'Gzpi:Q0': {('H', 'Z'): 0.01, ('H', 'X'): 0.008, ('S', 'X'): 0.012, ('A', 'X', 'Z'): 0.003},
                 'Gypi2:Q0': {('H', 'Y'): 0.004, ('H', 'X'): 0.003, ('S', 'X'): 0.006, ('S', 'Z'): 0.005}}
        model = create_explicit_model(pspec, lindblad_error_coeffs=noise,
                                      lindblad_parameterization='GLND', simulator='matrix')
        for ref in ('Gzpi', 'Gypi2'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                A, info = cgstgauge.standard_gauge_transform(model, target, ref, return_info=True)
            self.assertEqual(len(info['spam_fixed_basis']), 2)
            self.assertTrue(any('commute with every ideal gate' in str(w.message) for w in caught))
            self.assertTrue(np.all(np.isfinite(A)))

    def test_spam_gauge_off(self):
        # with the SPAM tie-break disabled, the SPAM-only direction is left at the identity
        A, info = cgstgauge.standard_gauge_transform(self.model, self.target, 'Gzpi2', ['Gypi2'],
                                                     return_info=True, fix_spam_gauge=False)
        self.assertArraysAlmostEqual(info['spam_params'], np.zeros(1))
        self.assertIsNone(info['spam_optimizer_result'])
        self.assertLess(np.abs(A - np.identity(4)).max(), 1e-4)


class ErrorgenCoefficientsTester(BaseCase):

    def test_round_trip(self):
        pspec = _pspec()
        target = create_explicit_model(pspec, simulator='matrix')
        coeffs = {'Gzpi2:Q0': {('H', 'Z'): 0.01, ('S', 'X'): 0.002, ('C', 'X', 'Y'): 0.001, ('A', 'X', 'Z'): 0.0005},
                  'Gypi2:Q0': {('H', 'Y'): -0.004, ('S', 'Z'): 0.003, ('C', 'Y', 'Z'): -0.0007, ('A', 'X', 'Y'): 0.0012}}
        model = create_explicit_model(pspec, lindblad_error_coeffs=coeffs,
                                      lindblad_parameterization='GLND', simulator='matrix')
        out = cgstgauge.errorgen_coefficients_in_gauge(model, target)
        self.assertEqual(set(out.keys()), set(target.operations.keys()))
        # keys are the full GLND label set (as used by errorgen_coefficient_labels)
        glnd = target.copy(); glnd.set_all_parameterizations('GLND')
        full_labels = glnd.operations[('Gzpi2', 'Q0')].errorgen_coefficient_labels()
        for lbl in out:
            self.assertEqual(set(out[lbl].keys()), set(full_labels))
            self.assertTrue(all(isinstance(k, GlobalElementaryErrorgenLabel) for k in out[lbl]))
        for gstr, d in coeffs.items():
            lbl = Label(*gstr.split(':'))
            expected = {GlobalElementaryErrorgenLabel(k[0], tuple(k[1:]), ('Q0',)): v for k, v in d.items()}
            for key, val in out[lbl].items():
                self.assertAlmostEqual(val, expected.get(key, 0.0), places=9, msg=str(key))
        self.assertTrue(all(abs(v) < 1e-12 for v in out[Label('Gi', 'Q0')].values()))
        # restricting the types just filters (the elementary generators form a dual basis)
        hs = cgstgauge.errorgen_coefficients_in_gauge(model, target, errorgen_types=('H', 'S'))
        for lbl in hs:
            self.assertEqual(set(k.errorgen_type for k in hs[lbl]), {'H', 'S'})
            for k, v in hs[lbl].items():
                self.assertAlmostEqual(v, out[lbl][k], places=12)
        # agrees with the model's own coefficient accounting
        own = model.operations[('Gzpi2', 'Q0')].errorgen_coefficients()
        for k, v in own.items():
            self.assertAlmostEqual(out[Label('Gzpi2', 'Q0')][GlobalElementaryErrorgenLabel.cast(k)], v, places=9)
