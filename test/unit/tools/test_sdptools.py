import inspect

import numpy as np

import pygsti.tools.basistools as bt
import pygsti.tools.sdptools as sdps
from pygsti.tools.jamiolkowski import jamiolkowski_iso

from ..util import BaseCase, needs_cvxpy


def _projective_instrument_members(basis):
    """
    Superoperators for the 1-qubit projective instrument E_i(rho) = P_i rho P_i,
    P_i = |i><i|, expressed in `basis`.  Their sum is the (CPTP) dephasing channel.
    """
    members = []
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        superop_std = np.kron(P, P.conj())
        members.append(bt.change_basis(superop_std, 'std', basis).real)
    return members


def _choi_min_eig(superop, basis):
    J = jamiolkowski_iso(superop, basis, basis, normalized=True)
    return np.min(np.linalg.eigvalsh(J))


def _tp_violation(superop_sum, basis):
    """max-abs violation of the TP condition S.T @ vec(I) == vec(I) (any basis)."""
    vecI = bt.stdmx_to_vec(np.eye(2), basis)
    return np.max(np.abs(superop_sum.T @ vecI - vecI))


class InstrumentProjectionTester(BaseCase):

    def setUp(self):
        self.members_pp = _projective_instrument_members('pp')
        # Perturb so that the members are outside the feasible set:
        # member 0 is scaled (so the sum is no longer TP) and member 1 gets a
        # negative multiple of member 0 added (so it is no longer CP, since the
        # two members' Choi matrices are rank-1 with orthogonal supports).
        self.perturbed_pp = [1.25 * self.members_pp[0],
                             self.members_pp[1] - 0.1 * self.members_pp[0]]

    def test_perturbed_members_are_infeasible(self):
        self.assertLess(_choi_min_eig(self.perturbed_pp[1], 'pp'), -1e-3)
        self.assertGreater(_tp_violation(sum(self.perturbed_pp), 'pp'), 1e-3)
        # ... while the unperturbed members are feasible.
        for member in self.members_pp:
            self.assertGreater(_choi_min_eig(member, 'pp'), -1e-12)
        self.assertLess(_tp_violation(sum(self.members_pp), 'pp'), 1e-12)

    @needs_cvxpy
    def test_frobenius_projection_fixes_feasible_instrument(self):
        projected, objective_val = sdps.project_instrument_members(self.members_pp, 'pp', norm='frobenius')
        self.assertAlmostEqual(objective_val, 0.0, places=5)
        for X, G in zip(projected, self.members_pp):
            self.assertArraysAlmostEqual(X, G, places=4)

    @needs_cvxpy
    def test_projection_of_perturbed_instrument(self):
        for norm in sdps.INSTRUMENT_PROJECTION_NORMS:
            with self.subTest(norm=norm):
                projected, objective_val = sdps.project_instrument_members(self.perturbed_pp, 'pp', norm=norm)
                self.assertTrue(np.isfinite(objective_val))
                self.assertGreater(objective_val, 1e-3)
                for X in projected:
                    self.assertIsNotNone(X)
                    self.assertGreater(_choi_min_eig(X, 'pp'), -1e-6)
                self.assertLess(_tp_violation(sum(projected), 'pp'), 1e-6)

    @needs_cvxpy
    def test_projection_in_non_identity_first_basis(self):
        # The 'std' basis has first_element_is_identity == False, exercising the
        # explicit S.T @ vec(I) == vec(I) form of the TP-sum constraint.  (This
        # fixture's std-basis superops happen to be real, as the model requires.)
        members_std = _projective_instrument_members('std')
        perturbed_std = [1.25 * members_std[0], members_std[1] - 0.1 * members_std[0]]
        projected, objective_val = sdps.project_instrument_members(perturbed_std, 'std', norm='frobenius')
        self.assertTrue(np.isfinite(objective_val))
        for X in projected:
            self.assertGreater(_choi_min_eig(X, 'std'), -1e-6)
        self.assertLess(_tp_violation(sum(projected), 'std'), 1e-6)

    def test_default_norm_is_frobenius(self):
        # The Frobenius norm is the unique-minimizer Euclidean projection and the
        # documented default; instrument-seeding code relies on it.
        for fn in (sdps.instrument_projection_model, sdps.project_instrument_members):
            self.assertEqual(inspect.signature(fn).parameters['norm'].default, 'frobenius')

    @needs_cvxpy
    def test_model_builder_contract(self):
        problem, member_vars = sdps.instrument_projection_model(self.perturbed_pp, 'pp', norm='diamond')
        self.assertEqual([X.name() for X in member_vars], ['X0', 'X1'])
        for X in member_vars:
            self.assertIn(X.name(), problem.var_dict)
        with self.assertRaises(ValueError):
            sdps.instrument_projection_model(self.perturbed_pp, 'pp', norm='trace')
        with self.assertRaises(ValueError):
            sdps.instrument_projection_model([], 'pp')
        with self.assertRaises(ValueError):
            sdps.instrument_projection_model([self.perturbed_pp[0], np.eye(9)], 'pp')
        with self.assertRaises(ValueError):
            complex_member = self.perturbed_pp[1] + 0.1j * np.eye(4)
            sdps.instrument_projection_model([self.perturbed_pp[0], complex_member], 'pp')
