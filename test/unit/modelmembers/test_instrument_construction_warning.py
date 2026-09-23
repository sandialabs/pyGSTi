import warnings

import numpy as np

import pygsti.tools.basistools as bt
import pygsti.tools.optools as ot
from pygsti.baseobjs import Basis
from pygsti.modelmembers.instruments import Instrument, diagnose_instrument
from ..util import BaseCase


def _sk(mat, basis):
    return bt.stdmx_to_vec(mat, basis).ravel().real


def _projective_members(basis):
    members = {}
    for k in range(2):
        P = np.zeros((2, 2))
        P[k, k] = 1.0
        members[f'p{k}'] = bt.change_basis(np.kron(P, P.conj()), 'std', basis).real
    return members


def _singular_base_warnings(ws):
    return [w for w in ws if 'post-measurement gate' in str(w.message)
            and 'singular' in str(w.message)]


class SingularBaseWarningTester(BaseCase):
    """Constructing a Lindblad instrument around a singular static base gate must
    warn (naming the outcome and the rank-cap consequence); a full-rank base must
    not.  This is the one-line diagnostic that identifies the frozen-base rank cap
    at construction time instead of after a failed fit."""

    def setUp(self):
        self.pp = Basis.cast('pp', 4)

    def test_projective_members_warn(self):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            Instrument.from_cptr_superops(_projective_members(self.pp), self.pp)
        found = _singular_base_warnings(ws)
        self.assertEqual(len(found), 2)   # one per singular outcome
        msg = str(found[0].message)
        self.assertIn('rank', msg)
        self.assertIn('patch_instrument_seed', msg)

    def test_full_rank_members_do_not_warn(self):
        E0 = np.diag([0.85, 0.15]).astype(complex)
        members = {'p0': ot.rootconj_superop(_sk(E0, self.pp), self.pp),
                   'p1': ot.rootconj_superop(_sk(np.eye(2) - E0, self.pp), self.pp)}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            Instrument.from_cptr_superops(members, self.pp)
        self.assertEqual(len(_singular_base_warnings(ws)), 0)

    def test_explicit_singular_gate_via_from_effects_warns(self):
        E0 = np.diag([0.85, 0.15]).astype(complex)
        effects = {'p0': _sk(E0, self.pp), 'p1': _sk(np.eye(2) - E0, self.pp)}
        G = np.outer(_sk(np.array([[1, 0], [0, 0]], complex), self.pp),
                     _sk(np.eye(2), self.pp)).real   # measure-and-reset, rank 1
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            Instrument.from_effects({k: (effects[k], G) for k in effects}, self.pp)
        found = _singular_base_warnings(ws)
        self.assertEqual(len(found), 2)
        self.assertIn('If this rank structure is intended', str(found[0].message))

    def test_full_tp_parameterization_does_not_warn(self):
        # 'full TP' gates are additively parameterized -- no frozen base, no cap.
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            Instrument.from_cptr_superops(_projective_members(self.pp), self.pp,
                                          gate_parameterization='full TP')
        self.assertEqual(len(_singular_base_warnings(ws)), 0)

    def test_diagnostics_scratch_host_does_not_duplicate(self):
        # diagnose_instrument(warn=False) hosts the deficient base internally;
        # the construction-time warning must not leak through it.
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            diagnose_instrument(_projective_members(self.pp), self.pp, warn=False)
        self.assertEqual(len(_singular_base_warnings(ws)), 0)
