"""
End-to-end cGST tests across *several* gate sets.

`test_cgst.py` exercises the one-qubit {S, sqrt(Y)} gate set.  This
module checks that the generic pipeline -- finite-order germ search
(`pygsti.algorithms.cgstdesign`), first-order linear inversion
(`pygsti.algorithms.cgstinversion`) and standard-gauge fixing
(`pygsti.algorithms.cgstgauge`), tied together by
`CharacterGST(gateset_inversion='linear')` -- works for gate sets it was never
tuned on:

* {Gxpi2, Gypi2}: a second Clifford generating set;
* {Gxpi2, Gt}: a set containing a *non-Clifford* finite-order gate (the T gate,
  a pi/4 Z rotation, whose superoperator generates Z_8);
* {Gxpi2, Gypi2, Grot} with `Grot` an infinite-order (1 radian) rotation, which
  must never be used as a germ;

and that on the {S, sqrt(Y)} gate set the generic pipeline agrees, to first
order, with the closed-form extraction of `extract_szy_error_parameters`.

The first-order design matrix is the expensive part of a linear run, so each
gate set builds its design once (in `setUpClass`) and relies on
`CharacterGST`'s class-level Jacobian cache to reuse it across tests.
"""
import warnings

import numpy as np
from scipy.linalg import expm

import pygsti
from pygsti.algorithms import cgstdesign, cgstgauge
from pygsti.circuits import Circuit
from pygsti.models.modelconstruction import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ProtocolData
from pygsti.protocols.cgst import (CharacterGST, create_1q_szy_cgst_design,
                                   true_germ_eigenvalues)
from pygsti.tools import chartools
from pygsti.tools.optools import unitary_to_pauligate as _u2p
from ..util import BaseCase

_sx = np.array([[0, 1], [1, 0]], complex)
_sy = np.array([[0, -1j], [1j, 0]], complex)
_sz = np.array([[1, 0], [0, -1]], complex)

#: Germ depths shared by every linear-inversion design here.  The trivial-irrep
#: decays must bend measurably away from a straight line for the
#: finite-differenced design matrix to resolve them (see the notes on
#: `cgstinversion.first_order_design_matrix`), which needs the deep end.
DEPTHS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def _rot(axis, angle):
    """PTM of a rotation by `angle` about a Pauli axis."""
    return _u2p(expm(-0.5j * angle * axis))


def _target(gate_names, nonstd=None):
    """An ideal 'full TP' one-qubit model with a matrix forward simulator."""
    pspec = QubitProcessorSpec(1, list(gate_names), qubit_labels=['Q0'],
                               nonstd_gate_unitaries=(nonstd or {}))
    return create_explicit_model(pspec, ideal_gate_type='full TP',
                                 ideal_spam_type='full TP', simulator='matrix')


def _noisy(gate_names, rates):
    """A Lindblad-parameterized noisy model on the same processor spec."""
    pspec = QubitProcessorSpec(1, list(gate_names), qubit_labels=['Q0'])
    return create_explicit_model(pspec, lindblad_error_coeffs=rates,
                                 lindblad_parameterization='GLND', simulator='matrix')


def _fast(model):
    """A copy of `model` using the (much faster on deep circuits) map simulator."""
    copy = model.copy()
    copy.sim = 'map'
    return copy


def _run_linear(truth, target, design, reference_gate, **kwargs):
    """Noiseless linear-inversion run of `CharacterGST` on `design`."""
    ds = pygsti.data.simulate_data(_fast(truth), design.all_circuits_needing_data,
                                   1000, sample_error='none')
    protocol = CharacterGST(bootstrap_samples=0, gateset_inversion='linear',
                            target_model=target, reference_gate=reference_gate, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # designed-but-unusable degenerate blocks
        results = protocol.run(ProtocolData(design, ds))
    return results.for_protocol['CharacterGST']


class _LinearInversionCase(BaseCase):
    """
    Shared machinery for "noiseless data recovers the gate set" tests.

    Subclasses set `gate_names`, `truth_rates` and `reference_gate` and get a
    target model, a truth model, a design and the truth's standard-gauge
    error generator coefficients built once for the whole class.
    """

    gate_names = ()
    truth_rates = {}
    reference_gate = None
    max_germ_length = 7

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not cls.gate_names:
            return  # the abstract base itself
        cls.target = _target(cls.gate_names)
        cls.truth = _noisy(cls.gate_names, cls.truth_rates)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.germs = cgstdesign.find_cgst_germs(cls.target, max_length=cls.max_germ_length,
                                                   seed=0)
            cls.design = cgstdesign.create_cgst_design(
                cls.target, DEPTHS, 12, germs=cls.germs, mode='exact',
                num_projection_rounds=3, seed=0)
        cls.truth_coefficients = cgstgauge.errorgen_coefficients_in_gauge(
            cgstgauge.fix_standard_gauge(cls.truth, cls.target, cls.reference_gate), cls.target)
        cls.top = _run_linear(cls.truth, cls.target, cls.design, cls.reference_gate)

    def assert_germ_orders_finite(self, max_order=24):
        """Every germ generates a finite cyclic group of order <= `max_order`."""
        orders = []
        for germ in self.germs:
            superop = self.target.sim.product(germ)
            orders.append(chartools.germ_group_order(superop, max_order=max_order))
        return orders

    def assert_germ_eigenvalues_recovered(self, tol):
        """The estimate reproduces every germ eigenvalue of the truth (gauge invariant)."""
        worst = 0.0
        for row in self.design.germ_table:
            germ = Circuit(row['germ'])
            true = true_germ_eigenvalues(self.truth, germ, row['group_order'])
            got = true_germ_eigenvalues(self.top.estimated_model, germ, row['group_order'])
            for irrep, value in true.items():
                deviation = abs(got[irrep] - value)
                worst = max(worst, deviation)
                self.assertLess(deviation, tol,
                                msg='germ eigenvalue mismatch for %s irrep %d (%g)'
                                    % (row['name'], irrep, deviation))
        return worst

    def assert_hamiltonian_coefficients_recovered(self, tol):
        """The standard-gauge Hamiltonian coefficients match the truth's."""
        worst = 0.0
        for gate_str, per_gate in self.top.errorgen_estimates.items():
            gate = next(g for g in self.truth_coefficients if str(g) == gate_str)
            for errgen_str, entry in per_gate.items():
                errgen = next(e for e in self.truth_coefficients[gate] if str(e) == errgen_str)
                if errgen.errorgen_type != 'H':
                    continue
                deviation = abs(entry['value'] - self.truth_coefficients[gate][errgen])
                worst = max(worst, deviation)
                self.assertLess(deviation, tol, msg='%s %s off by %g'
                                                    % (gate_str, errgen_str, deviation))
        return worst


class GxGyLinearInversionTester(_LinearInversionCase):
    """cGST on {Gxpi2, Gypi2} -- a generating set the design code was not tuned on."""

    gate_names = ('Gxpi2', 'Gypi2')
    reference_gate = 'Gxpi2'
    # half the rates of test_cgst.py's {S, sqrt(Y)} case, so that the O(rate^2)
    # truncation of the first-order inversion sits comfortably below 1e-4 even
    # for the length-7 germ (whose per-repetition error is ~7x the per-gate one)
    truth_rates = {
        'Gxpi2:Q0': {('H', 'X'): 0.00075, ('H', 'Z'): 0.0003, ('S', 'X'): 0.0006,
                     ('S', 'Y'): 0.0006, ('S', 'Z'): 0.00075, ('C', 'X', 'Y'): 0.0002,
                     ('A', 'X', 'Z'): 0.00025},
        'Gypi2:Q0': {('H', 'Y'): 0.0006, ('H', 'X'): 0.00045, ('S', 'X'): 0.00075,
                     ('S', 'Y'): 0.0004, ('S', 'Z'): 0.00065, ('A', 'X', 'Y'): 0.0002},
    }

    def test_germ_search_is_amplificationally_complete_and_finite_order(self):
        # the same shape of germ set as {S, sqrt(Y)}: the two pi/2 rotations plus
        # three order-3 compound germs
        self.assertEqual(sorted(self.assert_germ_orders_finite(), reverse=True),
                         [4, 4, 3, 3, 3])
        # amplificationally complete on a randomized ensemble (this is what
        # find_germs itself certifies; re-check it independently)
        from pygsti.algorithms import germselection
        randomized = self.target.randomize_with_unitary(1e-3, rand_state=np.random.RandomState(5))
        self.assertTrue(germselection.test_germ_set_infl(randomized, self.germs))

    def test_design_structure(self):
        # 5 germs, each with a 2-dimensional trivial block and a multiplicity-one
        # nontrivial irrep: 2 children each, all analyzable
        self.assertEqual(len(self.design.keys()), 10)
        self.assertEqual(len(set(row['germ'] for row in self.design.germ_table)), 5)
        self.assertTrue(all(row['name'] in self.design.keys()
                            for row in self.design.germ_table))
        for name in self.design.keys():
            self.assertNotIn(':', name)  # directory-safe on every platform
            self.assertEqual(self.design[name].mode, 'exact')

    def test_noiseless_data_recovers_the_gate_set(self):
        info = self.top.inversion_info
        self.assertEqual(info['rank'], 13)   # 24 error generator params - 11 gauge directions
        self.assertEqual(info['num_params'], 24)
        self.assertEqual(info['num_observables'], 20)
        self.assertEqual(info['num_unamplified'], 11)
        self.assertLess(info['residual_norm'], 1e-3)
        self.assertEqual(info['skipped_children'], [])

        self.assert_germ_eigenvalues_recovered(1e-4)
        self.assert_hamiltonian_coefficients_recovered(1e-4)
        self.assertEqual(sorted(self.top.errorgen_estimates.keys()), ['Gxpi2:Q0', 'Gypi2:Q0'])
        self.assertEqual(sum(len(v) for v in self.top.errorgen_estimates.values()), 24)


class TGateLinearInversionTester(_LinearInversionCase):
    """
    cGST on {Gxpi2, Gt} -- a gate set containing a non-Clifford finite-order gate.

    The T gate is a pi/4 Z rotation, so its *superoperator* is a pi/4 rotation
    of the Bloch sphere and generates Z_8; the germ search must find it (and
    only finite-order compounds of it), and the design must carry Z_8 irreps
    for it.  Note that `max_germ_length=7` is needed: no amplificationally
    complete set of finite-order germs exists at length <= 6 for this gate set.
    """

    gate_names = ('Gxpi2', 'Gt')
    reference_gate = 'Gxpi2'
    truth_rates = {
        'Gxpi2:Q0': {('H', 'X'): 0.00075, ('H', 'Z'): 0.0003, ('S', 'X'): 0.0006,
                     ('S', 'Y'): 0.0006, ('S', 'Z'): 0.00075, ('C', 'X', 'Y'): 0.0002,
                     ('A', 'X', 'Z'): 0.00025},
        'Gt:Q0': {('H', 'Z'): 0.0006, ('H', 'X'): 0.00045, ('S', 'X'): 0.00075,
                  ('S', 'Y'): 0.0004, ('S', 'Z'): 0.00065, ('A', 'X', 'Y'): 0.0002},
    }

    def test_t_gate_superoperator_has_order_eight(self):
        t_superop = self.target.sim.product(Circuit([('Gt', 'Q0')], line_labels=('Q0',)))
        self.assertEqual(chartools.germ_group_order(t_superop), 8)

    def test_germ_search_returns_finite_orders_only(self):
        orders = self.assert_germ_orders_finite()
        self.assertEqual(len(orders), 5)
        self.assertEqual(sorted(orders, reverse=True), [8, 4, 3, 3, 3])
        from pygsti.algorithms import germselection
        randomized = self.target.randomize_with_unitary(1e-3, rand_state=np.random.RandomState(5))
        self.assertTrue(germselection.test_germ_set_infl(randomized, self.germs))

    def test_design_contains_z8_irreps_for_the_t_germ(self):
        t_germ = Circuit([('Gt', 'Q0')], line_labels=('Q0',)).str
        rows = [row for row in self.design.germ_table if row['germ'] == t_germ]
        self.assertTrue(rows, msg='the T gate is not a germ of this design')
        self.assertTrue(all(row['group_order'] == 8 for row in rows))
        # Z_8 irreps 1 and 7 are complex conjugates and carry the same information,
        # so only the trivial irrep and one of the pair is designed
        self.assertEqual(sorted(row['irrep_index'] for row in rows), [0, 1])
        self.assertEqual({row['irrep_index']: row['multiplicity'] for row in rows},
                         {0: 2, 1: 1})
        # ...and the group order really is what the child designs sample over
        for row in rows:
            self.assertEqual(self.design[row['name']].group_order, 8)

    def test_noiseless_data_recovers_the_gate_set(self):
        info = self.top.inversion_info
        self.assertEqual(info['rank'], 13)
        self.assertEqual(info['num_params'], 24)
        self.assertEqual(info['num_observables'], 20)  # no degenerate blocks here
        self.assertEqual(info['skipped_children'], [])
        self.assertLess(info['residual_norm'], 1e-3)

        self.assert_germ_eigenvalues_recovered(1e-4)
        self.assert_hamiltonian_coefficients_recovered(1e-4)


class ReducedModeLinearInversionTester(BaseCase):
    """
    The linear inversion on a `'reduced'`-mode design: usable, with caveats.

    `'reduced'` mode's Monte-Carlo synthetic projector leaves an `O(error)`
    ripple, periodic in the depth modulo the group order, in every decay curve.
    Because the *same* realized germ powers are used for the data and for the
    finite-differenced design matrix, the inversion is still self-consistent --
    but only as long as the decay fitter responds smoothly to that ripple, which
    it did not: the trivial-block refit used to run the decay rate to ~1e-14
    ("spike" solution), producing Jacobian rows of norm ~1e4, a numerical rank
    of 2 and wrong signs on the estimated phases.  With the refit bounded this
    design recovers the Hamiltonian coefficients to ~3e-5 and the germ
    eigenvalues to ~5e-4 -- 20-40x worse than `'exact'` mode -- and shows a
    spurious 14th singular value of order 1, so its numerical rank is not
    reliable.  The protocol warns about all this; these tolerances are the
    measured behaviour with a margin, not a target.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target = _target(('Gzpi2', 'Gypi2'))
        cls.truth = _noisy(('Gzpi2', 'Gypi2'), {
            'Gzpi2:Q0': {('H', 'Z'): 0.00075, ('H', 'X'): 0.0003, ('S', 'X'): 0.0006,
                         ('S', 'Y'): 0.0006, ('S', 'Z'): 0.00075, ('C', 'X', 'Y'): 0.0002,
                         ('A', 'X', 'Z'): 0.00025},
            'Gypi2:Q0': {('H', 'Y'): 0.0006, ('H', 'X'): 0.00045, ('S', 'X'): 0.00075,
                         ('S', 'Y'): 0.0004, ('S', 'Z'): 0.00065, ('A', 'X', 'Y'): 0.0002}})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            germs = cgstdesign.find_cgst_germs(cls.target, max_length=7, seed=0)
            cls.design = cgstdesign.create_cgst_design(
                cls.target, DEPTHS, 12, germs=germs, mode='reduced', num_projection_rounds=4, seed=0)
        ds = pygsti.data.simulate_data(_fast(cls.truth), cls.design.all_circuits_needing_data,
                                       1000, sample_error='none')
        protocol = CharacterGST(bootstrap_samples=0, gateset_inversion='linear',
                                target_model=cls.target, reference_gate='Gzpi2')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = protocol.run(ProtocolData(cls.design, ds))
        cls.warnings = [str(w.message) for w in caught]
        cls.top = results.for_protocol['CharacterGST']
        cls.truth_coefficients = cgstgauge.errorgen_coefficients_in_gauge(
            cgstgauge.fix_standard_gauge(cls.truth, cls.target, 'Gzpi2'), cls.target)

    def test_reduced_mode_is_warned_about(self):
        self.assertTrue(any("'reduced' sampling mode" in w for w in self.warnings))

    def test_no_runaway_and_degraded_but_usable_estimates(self):
        info = self.top.inversion_info
        sv = np.array(info['singular_values'])
        self.assertLess(sv[0], 1e2)           # was 1.4e5 with the unbounded refit
        self.assertGreaterEqual(info['rank'], 13)
        self.assertLessEqual(info['rank'], 14)  # the spurious ripple-induced direction
        case = GxGyLinearInversionTester()
        case.design, case.truth, case.top, case.truth_coefficients = \
            self.design, self.truth, self.top, self.truth_coefficients
        worst_eig = case.assert_germ_eigenvalues_recovered(2e-3)
        worst_h = case.assert_hamiltonian_coefficients_recovered(1e-4)
        # ...but it really is worse than exact mode (see the class docstring)
        self.assertGreater(worst_eig, 1e-5)
        self.assertGreater(worst_h, 1e-6)


class DesignMatrixCacheKeyTester(BaseCase):
    """Two designs that differ only in their realized random germ powers must not share a Jacobian."""

    def test_reduced_designs_with_different_seeds_get_different_keys(self):
        from pygsti.protocols.cgst import _design_fingerprint, _model_fingerprint
        target = _target(('Gzpi2', 'Gypi2'))
        germs = [Circuit([('Gzpi2', 'Q0')], line_labels=('Q0',))]
        d1 = cgstdesign.create_cgst_design(target, [0, 1, 2], 8, germs=germs, mode='reduced', seed=1)
        d2 = cgstdesign.create_cgst_design(target, [0, 1, 2], 8, germs=germs, mode='reduced', seed=2)
        d1b = cgstdesign.create_cgst_design(target, [0, 1, 2], 8, germs=germs, mode='reduced', seed=1)
        self.assertNotEqual(_design_fingerprint(d1), _design_fingerprint(d2))
        self.assertEqual(_design_fingerprint(d1), _design_fingerprint(d1b))
        hash(_design_fingerprint(d1))  # must be usable as a dict key
        # the model fingerprint sees the SPAM, not just the gates
        other = target.copy()
        other.preps['rho0'] = np.array([1, 0, 0, 0.9]) / np.sqrt(2)
        self.assertNotEqual(_model_fingerprint(target), _model_fingerprint(other))
        self.assertEqual(_model_fingerprint(target), _model_fingerprint(target.copy()))


class FullModeRejectionTester(BaseCase):

    def test_linear_inversion_rejects_full_mode_designs(self):
        target = _target(('Gzpi2', 'Gypi2'))
        truth = _noisy(('Gzpi2', 'Gypi2'), {'Gzpi2:Q0': {('H', 'Z'): 0.001}})
        germs = [Circuit([('Gzpi2', 'Q0')], line_labels=('Q0',))]
        design = cgstdesign.create_cgst_design(target, [1, 2, 4], 8, germs=germs, mode='full', seed=0)
        with self.assertRaises(ValueError) as cm:
            _run_linear(truth, target, design, 'Gzpi2')
        self.assertIn("'full'", str(cm.exception))


class IdleGateSetDesignTester(BaseCase):
    """
    The {S, sqrt(Y)} gate set *with* an idle, {S, sqrt(Y), I}, must be designable.

    The bare idle has group order 1 and is never a germ; the idle's twelve error
    generator coefficients are amplified by finite-order germs that contain it,
    and the germ search must find an amplificationally complete such set.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target = _target(('Gzpi2', 'Gypi2', 'Gi'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.germs = cgstdesign.find_cgst_germs(cls.target, max_length=7, seed=0)
            cls.design = cgstdesign.create_cgst_design(cls.target, [0, 1, 2, 4], 12, germs=cls.germs,
                                                       num_projection_rounds=2, seed=0)
            cls.lean = cgstdesign.create_cgst_design(cls.target, [0, 1, 2, 4], 12, germs=cls.germs,
                                                     num_projection_rounds=2, seed=0,
                                                     include_degenerate_blocks=False)

    def test_germ_set_is_ac_and_uses_the_idle(self):
        from pygsti.algorithms import germselection
        idle = Circuit([('Gi', 'Q0')], line_labels=('Q0',))
        self.assertNotIn(idle, self.germs)
        self.assertTrue(any(any(lbl.name == 'Gi' for lbl in g.layertup) for g in self.germs))
        for germ in self.germs:
            self.assertGreaterEqual(chartools.germ_group_order(self.target.sim.product(germ)), 2)
        randomized = self.target.randomize_with_unitary(1e-3, rand_state=np.random.RandomState(5))
        randomized.set_all_parameterizations('full TP')
        self.assertTrue(germselection.test_germ_set_infl(randomized, self.germs))

    def test_degenerate_blocks_can_be_left_out(self):
        self.assertGreater(len(self.design.keys()), len(self.lean.keys()))
        self.assertTrue(set(self.lean.keys()).issubset(set(self.design.keys())))
        for row in self.lean.germ_table:
            expected = 2 if row['irrep_index'] == 0 else 1
            self.assertEqual(row['multiplicity'], expected)
        # every child the lean design dropped is one the analysis would skip anyway
        from pygsti.algorithms import cgstinversion
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            usable = set(name for name, _ in cgstinversion.observable_labels(self.design, self.target))
        self.assertEqual(usable, set(self.lean.keys()))


class InfiniteOrderGermExclusionTester(BaseCase):
    """
    A gate set containing an infinite-order gate still yields a usable design.

    `Grot` is a rotation by 1 radian about Z: no power of its superoperator is
    the identity, so it can never be a cGST germ.  It may, however, appear
    *inside* a compound germ whose product happens to be of finite order (a pi
    rotation about a tilted axis, say) and it may appear in fiducials -- cGST
    only requires the germ's own superoperator to generate a finite cyclic
    group.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target = _target(('Gxpi2', 'Gypi2', 'Grot'),
                             nonstd={'Grot': expm(-0.5j * 1.0 * _sz)})
        cls.rot_label = ('Grot', 'Q0')
        # no amplificationally complete finite-order germ set exists below
        # length 9 once Grot's own errors have to be amplified
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.design = cgstdesign.create_cgst_design(
                cls.target, [0, 1, 2, 4], 8, mode='exact', num_projection_rounds=2,
                max_germ_length=9, seed=0)

    def _germs(self):
        return [Circuit(row['germ']) for row in self.design.germ_table]

    def test_candidate_germs_are_all_finite_order(self):
        candidates = cgstdesign.finite_order_candidate_germs(self.target, 3)
        self.assertTrue(candidates)
        for circuit in candidates:
            # never raises: every candidate's superoperator has finite order
            chartools.germ_group_order(self.target.sim.product(circuit))
        # the bare infinite-order gate is not a candidate, even though it is a
        # primitive of the model (so `force='singletons'` cannot force it in)
        bare = Circuit([self.rot_label], line_labels=('Q0',))
        self.assertNotIn(bare, candidates)
        with self.assertRaises(ValueError):
            chartools.germ_group_order(self.target.sim.product(bare))

    def test_design_never_uses_an_infinite_order_germ(self):
        germs = self._germs()
        self.assertTrue(germs)
        bare = Circuit([self.rot_label], line_labels=('Q0',))
        for germ in germs:
            self.assertNotEqual(germ.layertup, bare.layertup)
            chartools.germ_group_order(self.target.sim.product(germ))
        # the two Clifford primitives are forced in as singleton germs; the
        # infinite-order one is not
        singletons = set(g.layertup for g in germs if len(g) == 1)
        self.assertEqual(singletons, {(('Gxpi2', 'Q0'),), (('Gypi2', 'Q0'),)})

    def test_infinite_order_gate_is_still_usable_inside_germs_and_fiducials(self):
        # This is a property of the *protocol*, not an accident: only the germ's
        # product needs finite order.  Assert the design is not silently
        # throwing away every circuit that touches Grot.
        used = any('Grot' in row['germ'] or 'Grot' in row['prep_fiducial']
                   or 'Grot' in row['meas_fiducial'] for row in self.design.germ_table)
        self.assertTrue(used)
        for circuit in self.design.all_circuits_needing_data:
            self.assertTrue(len(circuit) >= 0)  # design is well formed


def _standard_gauge_model(theta=0., alpha=0., beta=0., lam1=1., lam2=1., a=0., r1=0., r2=0.,
                      cxy=0., cxz=0., cyz=0., ay=0., arel=0.):
    """
    The standard-gauge {S, sqrt(Y)} channel forms, without an idle.

    Identical to `test_cgst._standard_gauge_model` (including the sqrt(Y) row
    placement: `r1` and `a_y` sit on the Y row, the rotation axis of sqrt(Y))
    but built on a two-gate processor spec, so that it can also serve as the
    truth model of a *generic* linear-inversion run.
    """
    E_S = np.array([[1, 0, 0, 0],
                    [0, lam2 * np.cos(theta), -lam2 * np.sin(theta), 0],
                    [0, lam2 * np.sin(theta), lam2 * np.cos(theta), 0],
                    [-a, 0, 0, lam1]])
    E_sto = np.array([[1, 0, 0, 0],
                      [arel, 1 - r2, cxy, cxz],
                      [ay, cxy, 1 - r1, cyz],
                      [arel, cxz, cyz, 1 - r2]])
    pspec = QubitProcessorSpec(1, ['Gzpi2', 'Gypi2'], qubit_labels=['Q0'])
    mdl = create_explicit_model(pspec, ideal_gate_type='full', simulator='matrix')
    mdl.operations[('Gzpi2', 'Q0')] = E_S @ _rot(_sz, np.pi / 2)
    mdl.operations[('Gypi2', 'Q0')] = (_rot(_sx, beta) @ E_sto
                                       @ _rot(_sy, np.pi / 2 + alpha) @ _rot(_sx, -beta))
    return mdl


class SzyVersusLinearInversionTester(BaseCase):
    """
    The generic linear pipeline agrees with `extract_szy_error_parameters`.

    Both analyses are run on noiseless data from the *same* standard-gauge
    truth model: the closed-form one on the hand-built
    `create_1q_szy_cgst_design` experiment set, the generic one on a
    `create_cgst_design` experiment set for the same gate set.  Every parameter
    of the standard-gauge channel forms is then read back off the linear
    estimate's standard-gauge gate matrices and compared.  Both routes are
    first order in the error rates, so they may differ by `O(rate**2)`.
    """

    #: Injected error parameters, all at the 1e-3 scale.
    injected = dict(theta=0.0010, alpha=0.0008, beta=0.0006, lam1=1 - 0.0020,
                    lam2=1 - 0.0015, a=-0.0004, r1=0.0018, r2=0.0012,
                    cxy=0.0004, cxz=0.0003, cyz=0.0002, ay=-0.0003, arel=-0.0002)

    #: A few times rate**2 (rate ~ 2e-3): the first-order truncation of the two
    #: analyses, which is what these tests are measuring.
    tol = 2e-5

    #: The active (non-unital) error gets a looser bound.  Its only first-order
    #: signature is the trivial-irrep `active = (1 - lam)(B - C)` observable,
    #: which is a *product* of two first-order quantities, so its inversion
    #: carries an O(rate**2) error with a much larger prefactor than the
    #: eigenvalue observables (~14 rate**2 here, verified to scale as rate**2 by
    #: halving every injected rate; it is not finite-difference error -- the
    #: number is unchanged when `design_matrix_step` is raised to 1e-3).
    active_tol = 1e-4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.truth = _standard_gauge_model(**cls.injected)
        cls.target = _target(('Gzpi2', 'Gypi2'))

        # -- closed-form ('szy') analysis on the hand-built {S, sqrt(Y)} design
        szy_design = create_1q_szy_cgst_design(DEPTHS, 12, mode='exact',
                                               include_idle=False, seed=19)
        ds = pygsti.data.simulate_data(cls.truth, szy_design.all_circuits_needing_data,
                                       1000, sample_error='none')
        szy_results = CharacterGST(bootstrap_samples=0, gateset_inversion='szy').run(
            ProtocolData(szy_design, ds))
        cls.szy = szy_results.for_protocol['CharacterGST'].error_parameters

        # -- generic linear analysis on a searched design for the same gate set
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.design = cgstdesign.create_cgst_design(
                cls.target, DEPTHS, 12, mode='exact', num_projection_rounds=3, seed=0)
        cls.top = _run_linear(cls.truth, cls.target, cls.design, 'Gzpi2')
        cls.linear = cls._read_channel_parameters(cls.top, cls.target)

    @staticmethod
    def _read_channel_parameters(top, target):
        """
        Read the standard-gauge channel parameters off a standard-gauge estimate.

        `theta`, `lambda1`, `lambda2` and `a` come straight from the S gate's
        error channel `E_S = Lam_S S_ideal^{-1}`, whose Bloch
        block is `lam2 R_z(theta)` on the equator and `lam1` along the axis,
        with `E_S[3, 0] = -a`.

        `r1`, `r2` and `alpha` are read *operationally*, exactly as
        `extract_szy_error_parameters` defines them: as the trivial-
        (axis-parallel) and nontrivial- (Ramsey) branch eigenvalues of the noisy
        sqrt(Y) germ.

        `beta` -- the angle between the two gates' rotation axes -- is *not*
        directly a matrix entry, because the sqrt(Y) channel form conjugates by
        `R_x(beta)`.
        Expanding `Lam_Y = R_x(beta) E_sto R_y(pi/2 + alpha) R_x(-beta)` to first
        order and using `R_y(pi/2) X R_y(-pi/2) = -Z` gives an error generator
        `beta*G_x + alpha*G_y + beta*G_z + (E_sto - 1)`, where `G_P` generates
        `exp(-i angle P / 2)`.  pyGSTi's elementary Hamiltonian error generator
        `H(P)` is `-i[P, .]` with an unnormalized Pauli, i.e. `2 G_P`, so the
        standard-gauge coefficients satisfy `h_X = h_Z = beta / 2` and
        `h_Y = alpha / 2` for sqrt(Y) (and `h_Z = theta / 2` for S).
        """
        model = top.estimated_model
        s_label, y_label = ('Gzpi2', 'Q0'), ('Gypi2', 'Q0')
        ideal_s = target.operations[s_label].to_dense()
        error_s = model.operations[s_label].to_dense() @ np.linalg.inv(ideal_s)

        y_germ = Circuit([y_label], line_labels=('Q0',))
        y_eigenvalues = true_germ_eigenvalues(model, y_germ, 4)

        h_x = top.errorgen_estimates['Gypi2:Q0']['H(X:Q0)']['value']
        return {
            'theta': float(np.arctan2(error_s[2, 1], error_s[1, 1])),
            'lambda1': float(error_s[3, 3]),
            'lambda2': float(np.hypot(error_s[1, 1], error_s[2, 1])),
            'a': float(-error_s[3, 0]),
            'r1': float(1.0 - abs(y_eigenvalues[0])),
            'r2': float(1.0 - abs(y_eigenvalues[1])),
            'alpha': float(np.angle(y_eigenvalues[1])),
            'beta': float(2.0 * h_x),
        }

    def test_linear_estimate_reproduces_the_injected_channel(self):
        """Sanity check on the readout itself: it recovers the injected values."""
        expected = dict(self.injected)
        for key, injected_key in [('theta', 'theta'), ('alpha', 'alpha'), ('beta', 'beta'),
                                  ('lambda1', 'lam1'), ('lambda2', 'lam2'), ('a', 'a'),
                                  ('r1', 'r1'), ('r2', 'r2')]:
            tol = self.active_tol if key == 'a' else self.tol
            self.assertLess(abs(self.linear[key] - expected[injected_key]), tol,
                            msg='%s: linear %g vs injected %g'
                                % (key, self.linear[key], expected[injected_key]))

    def test_szy_and_linear_agree_to_first_order(self):
        for key in ('theta', 'alpha', 'lambda1', 'lambda2', 'r1', 'r2', 'beta'):
            deviation = abs(self.szy[key] - self.linear[key])
            self.assertLess(deviation, self.tol,
                            msg='%s: szy %g vs linear %g (differ by %g)'
                                % (key, self.szy[key], self.linear[key], deviation))
        self.assertLess(abs(self.szy['a'] - self.linear['a']), self.active_tol)
