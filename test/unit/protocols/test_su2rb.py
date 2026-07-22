"""
Tests for pygsti.protocols.su2rb: SU(2) synthetic SPAM RB circuits and experiment
designs, the SU2QuditRBSimulator, the zero-noise variance diagnostics, and the
SU2QuditRB (R1RB) protocol.
"""
import tempfile
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning

from pygsti.circuits.circuit import Circuit
from pygsti.protocols.su2rb import (
    SU2QuditRBDesign,
    SU2QuditRBSimulator,
    SU2QuditRB,
    SU2QuditRBResults,
    predicted_zero_noise_variance,
    circuit_from_euler_angles,
    euler_angles_from_circuit,
    jz_dephasing,
    jz_rotation,
    compose_noise_channels,
    _add_spam_layers_inplace
)
from pygsti.protocols.protocol import ProtocolData
from pygsti.data.dataset import DataSet
from pygsti.tools.su2tools import SpinJ, distance_mod_phase, random_euler_angles, composition_inverse, GATE_NAME
from pygsti.tools.optools import unitary_to_std_process_mx
from pygsti.tools.exceptions import RBFitFailureWarning

from ..util import BaseCase


# ---------------------------------------------------------------------------
# circuit_from_euler_angles / euler_angles_from_circuit
# ---------------------------------------------------------------------------

class TestEulerAngleCircuitHelpers(BaseCase):

    def test_round_trip_no_prep(self):
        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 4 * np.pi, size=(5, 3))
        c = circuit_from_euler_angles(angles, qudit_label='Q0')
        c.done_editing()
        self.assertEqual(len(c), 5)
        for layer in c:
            self.assertEqual(layer.name, GATE_NAME)

        recovered = euler_angles_from_circuit(c)
        # Agreement to <= 1e-12 is required; empirically the string round trip is
        # bit-exact, so we check that too.
        self.assertTrue(np.allclose(recovered, angles, atol=1e-12))
        self.assertTrue(np.array_equal(recovered, angles))

    def test_round_trip_through_string(self):
        rng = np.random.default_rng(1)
        angles = rng.uniform(0, 4 * np.pi, size=(4, 3))
        c = circuit_from_euler_angles(angles, qudit_label='Q0')
        _add_spam_layers_inplace(c, prep_index=1)
        s = c.str
        c2 = Circuit(None, stringrep=s, line_labels=c.line_labels)
        self.assertEqual(c, c2)
        recovered = euler_angles_from_circuit(c2)
        self.assertTrue(np.array_equal(recovered, angles))

    def test_prep_and_povm_labels_embedded(self):
        angles = np.zeros((2, 3))
        c = circuit_from_euler_angles(angles, qudit_label='Q0')
        _add_spam_layers_inplace(c, prep_index=2)
        names = [layer.name for layer in c]
        self.assertEqual(names[0], 'rho2')
        self.assertEqual(names[-1], 'Mdefault')
        self.assertEqual(names.count(GATE_NAME), 2)

    def test_prep_index_out_of_range_raises(self):
        angles = np.zeros((1, 3))
        c = circuit_from_euler_angles(angles, qudit_label='Q0')
        with self.assertRaises(ValueError):
            _add_spam_layers_inplace(c, prep_index=2, j=0.5)  # dim=2, valid range 0,1

    def test_bad_angle_shape_raises(self):
        with self.assertRaises(ValueError):
            circuit_from_euler_angles(np.zeros((3, 2)))

    def test_euler_angles_from_circuit_empty_for_no_gu_layers(self):
        # A circuit with no `Gu` layers returns a `(0, 3)` array rather than raising.
        c = Circuit(['rho0', 'Mdefault'], line_labels=('Q0',))
        recovered = euler_angles_from_circuit(c)
        self.assertEqual(recovered.shape, (0, 3))


# ---------------------------------------------------------------------------
# SU2QuditRBDesign
# ---------------------------------------------------------------------------

def _net_unitary(spinj, angles):
    Us = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
    net = np.eye(spinj.dim, dtype=complex)
    for U in Us:
        net = U @ net
    return net


class _SU2RBDesignChecks:
    """Mixin providing the shared design-level tests, parameterized by spin `j`.
    `SU2QuditRBDesign` is the only design class, always hidden-first-gate, so there is
    no `design_cls`/extra-attrs axis to parameterize over -- just `j`."""

    j = None

    def test_circuit_counts(self):
        depths = [1, 2, 4]
        circuits_per_depth = 3
        design = SU2QuditRBDesign(self.j, depths, circuits_per_depth, seed=1234)
        dim = design.dim
        self.assertEqual(dim, round(2 * self.j) + 1)
        for circuits_at_depth in design.circuit_lists:
            self.assertEqual(len(circuits_at_depth), circuits_per_depth * dim)

    def test_aux_lists_aligned_with_circuits(self):
        depths = [1, 3]
        circuits_per_depth = 2
        design = SU2QuditRBDesign(self.j, depths, circuits_per_depth, seed=5)
        for depth_idx, circuits_at_depth in enumerate(design.circuit_lists):
            n = len(circuits_at_depth)
            self.assertEqual(len(design.euler_angles[depth_idx]), n)
            self.assertEqual(len(design.seq_index[depth_idx]), n)
            self.assertEqual(len(design.prep_index[depth_idx]), n)
            self.assertEqual(len(design.charcores[depth_idx]), n)
            for circuit_idx, c in enumerate(circuits_at_depth):
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                self.assertTrue(np.array_equal(angles, euler_angles_from_circuit(c)))
                prep_idx = design.prep_index[depth_idx][circuit_idx]
                names = [layer.name for layer in c]
                self.assertEqual(names[0], 'rho%d' % prep_idx)
                self.assertEqual(names[-1], 'Mdefault')

    def test_ideal_composition(self):
        # Every SU2QuditRBDesign is hidden-first-gate: the appended inverting gate only
        # cancels rows 1..m-1, so the net ideal composition of the full (m+1)-gate
        # sequence is the random first gate (row 0), not the identity.
        spinj = SpinJ(self.j)
        depths = [1, 2, 5]
        design = SU2QuditRBDesign(self.j, depths, circuits_per_depth=4, seed=7)
        for depth_idx, depth in enumerate(design.depths):
            # only need to check one prep's circuit per sequence -- the angle array is
            # shared across all `dim` preps generated from the same sequence.
            for seq in range(design.circuits_per_depth):
                circuit_idx = seq * design.dim  # prep_index == 0 circuit for this sequence
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                self.assertEqual(angles.shape, (depth + 1, 3))
                net = _net_unitary(spinj, angles)
                target = spinj.unitaries_from_angles(angles[0, 0], angles[0, 1], angles[0, 2])[0]
                self.assertLess(distance_mod_phase(net, target), 1e-10)

    def test_truncate_to_lists(self):
        depths = [1, 2, 3]
        design = SU2QuditRBDesign(self.j, depths, circuits_per_depth=2, seed=3)
        truncated = design.truncate_to_lists([0, 2])
        self.assertEqual(truncated.depths, [1, 3])
        self.assertEqual(len(truncated.circuit_lists), 2)
        self.assertEqual(truncated.circuit_lists[0], design.circuit_lists[0])
        self.assertEqual(truncated.circuit_lists[1], design.circuit_lists[2])
        self.assertEqual(truncated.euler_angles[0], design.euler_angles[0])
        self.assertEqual(truncated.euler_angles[1], design.euler_angles[2])
        self.assertEqual(truncated.seq_index[1], design.seq_index[2])
        self.assertEqual(truncated.prep_index[1], design.prep_index[2])
        self.assertEqual(truncated.charcores[0], design.charcores[0])
        self.assertEqual(truncated.charcores[1], design.charcores[2])

    def test_serialization_round_trip(self):
        depths = [1, 2]
        design = SU2QuditRBDesign(self.j, depths, circuits_per_depth=2, seed=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            design.write(tmpdir)
            reloaded = SU2QuditRBDesign.from_dir(tmpdir)
        self.assertEqual(design.circuit_lists, reloaded.circuit_lists)
        self.assertEqual(design.euler_angles, reloaded.euler_angles)
        self.assertEqual(design.seq_index, reloaded.seq_index)
        self.assertEqual(design.prep_index, reloaded.prep_index)
        self.assertEqual(design.charcores, reloaded.charcores)
        self.assertEqual(design.j, reloaded.j)
        self.assertEqual(design.dim, reloaded.dim)


class TestSU2RBDesignHalfInt(_SU2RBDesignChecks, BaseCase):
    j = 0.5


class TestSU2RBDesignThreeHalves(_SU2RBDesignChecks, BaseCase):
    j = 1.5


class TestSU2RBDesignExtras(BaseCase):
    """Design-specific checks that don't fit the shared per-j mixin."""

    def test_charcores_match_direct_evaluation(self):
        from pygsti.tools.su2tools import charactercores_from_euler_angles

        j = 1.5
        design = SU2QuditRBDesign(j, depths=[1, 4], circuits_per_depth=3, seed=42)
        dim = design.dim
        irrep_labels = np.arange(dim)
        for depth_idx in range(len(design.depths)):
            for circuit_idx in range(len(design.circuit_lists[depth_idx])):
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                first_gate_angles = angles[0, :].reshape((1, -1))
                expected_cores = charactercores_from_euler_angles(irrep_labels, first_gate_angles)
                actual_cores = np.array(design.charcores[depth_idx][circuit_idx]).reshape((1, -1))
                self.assertTrue(np.allclose(actual_cores, expected_cores, atol=1e-12))

    def test_depth_zero_raises(self):
        with self.assertRaises(ValueError):
            SU2QuditRBDesign(0.5, depths=[0, 1], circuits_per_depth=2, seed=0)


# ---------------------------------------------------------------------------
# SU2QuditRBSimulator
# ---------------------------------------------------------------------------

def _dataset_probs(dataset, circuit, dim):
    """Reconstruct a length-`dim` probability vector (indexed by effect index) from a
    DataSet row's raw ('none'-sample-error or multinomial) counts."""
    row = dataset[circuit]
    counts = row.counts
    total = sum(counts.values())
    return np.array([counts.get((str(ell),), 0.0) for ell in range(dim)]) / total


def _net_unitary_from_angles(spinj, angles):
    Us = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
    net = np.eye(spinj.dim, dtype=complex)
    for U in Us:
        net = U @ net
    return net


def _crossval_model(spinj, qudit_label='Q0'):
    """
    Build an ExplicitOpModel over an explicit single-qudit state space (dimension
    `spinj.dim`, constructed directly rather than via `default_space_for_udim`,
    which would auto-cast a dimension like 4 or 8 to a multi-qubit space instead of
    a single qudit) whose `Gu` factory reproduces `spinj.unitaries_from_angles`, and
    whose `rho{ell}`/`Mdefault` preps/POVM match the SU2QuditRBDesign circuit convention.
    Used to cross-validate SU2QuditRBSimulator's noiseless probabilities against
    pyGSTi's standard model.probabilities() path.
    """
    from pygsti.baseobjs import ExplicitStateSpace
    from pygsti.baseobjs.basis import BuiltinBasis
    from pygsti.tools.basistools import stdmx_to_vec
    from pygsti.models.explicitmodel import ExplicitOpModel
    from pygsti.modelmembers.povms import UnconstrainedPOVM
    from pygsti.modelmembers.states import FullState
    from pygsti.modelmembers.operations.opfactory import UnitaryOpFactory

    dim = spinj.dim
    state_space = ExplicitStateSpace([qudit_label], [dim])
    mx_basis = BuiltinBasis('gm', state_space.dim)  # generalized Gell-Mann: arbitrary-dim, real, Hermitian

    model = ExplicitOpModel(state_space, mx_basis)
    for ell in range(dim):
        rho = np.zeros((dim, dim), dtype=complex)
        rho[ell, ell] = 1.0
        model.preps['rho%d' % ell] = FullState(stdmx_to_vec(rho, mx_basis))

    effects = []
    for ell in range(dim):
        E = np.zeros((dim, dim), dtype=complex)
        E[ell, ell] = 1.0
        effects.append((str(ell), stdmx_to_vec(E, mx_basis)))
    model.povms['Mdefault'] = UnconstrainedPOVM(effects, evotype='default')

    def fn(args):
        alpha, beta, gamma = args
        return spinj.unitaries_from_angles(alpha, beta, gamma)[0]

    model.factories[GATE_NAME] = UnitaryOpFactory(fn, state_space, superop_basis=mx_basis, evotype='densitymx')
    return model


class TestSU2QuditRBSimulatorNoiseless(BaseCase):

    def test_hand_composed_probabilities(self):
        j = 1.5
        spinj = SpinJ(j)
        design = SU2QuditRBDesign(j, depths=[2, 4], circuits_per_depth=3, seed=1)
        sim = SU2QuditRBSimulator(spinj)
        self.assertTrue(sim.is_noiseless)
        data = sim.run(design)

        for depth_idx, circuits_at_depth in enumerate(design.circuit_lists):
            for circuit, angles, prep_idx in zip(
                    circuits_at_depth, design.euler_angles[depth_idx], design.prep_index[depth_idx]):
                net = _net_unitary_from_angles(spinj, np.array(angles))
                expected = np.abs(net[:, prep_idx]) ** 2
                actual = _dataset_probs(data.dataset, circuit, spinj.dim)
                self.assertTrue(np.allclose(actual, expected, atol=1e-10))

    def test_dataset_circuits_match_design(self):
        j = 0.5
        design = SU2QuditRBDesign(j, depths=[1, 2], circuits_per_depth=2, seed=2)
        sim = SU2QuditRBSimulator(j)
        data = sim.run(design)
        design_circuits = set(design.all_circuits_needing_data)
        dataset_circuits = set(data.dataset.keys())
        self.assertEqual(design_circuits, dataset_circuits)


class TestSU2QuditRBSimulatorCrossValidation(BaseCase):
    """model.probabilities(circuit) agreement, pinning the circuit label convention
    to the simulator's semantics."""

    def test_matches_model_probabilities_j_half(self):
        self._check_j(0.5)

    def test_matches_model_probabilities_j_three_halves(self):
        self._check_j(1.5)

    def _check_j(self, j):
        spinj = SpinJ(j)
        model = _crossval_model(spinj)
        design = SU2QuditRBDesign(j, depths=[1, 3], circuits_per_depth=2, seed=99)
        sim = SU2QuditRBSimulator(spinj)
        data = sim.run(design)

        for circuits_at_depth in design.circuit_lists:
            for circuit in circuits_at_depth:
                model_probs = model.probabilities(circuit)
                sim_probs = _dataset_probs(data.dataset, circuit, spinj.dim)
                for ell in range(spinj.dim):
                    self.assertAlmostEqual(sim_probs[ell], model_probs[(str(ell),)], places=12)


class TestSU2QuditRBSimulatorSpamSwap(BaseCase):

    def test_probabilities_from_compositions_matches_run(self):
        j = 1.5
        spinj = SpinJ(j)
        noise = jz_dephasing(spinj, 0.15, 1.0)
        sim = SU2QuditRBSimulator(spinj, noise_channel=noise)
        design = SU2QuditRBDesign(j, depths=[1, 3], circuits_per_depth=4, seed=3)
        data = sim.run(design)

        compositions = sim.compute_nonspam_compositions(design)
        probs_from_comp = sim.probabilities_from_compositions(compositions)

        for depth_idx, circuits_at_depth in enumerate(design.circuit_lists):
            seq_at_depth = design.seq_index[depth_idx]
            prep_at_depth = design.prep_index[depth_idx]
            for circuit, seq_idx, prep_idx in zip(circuits_at_depth, seq_at_depth, prep_at_depth):
                expected = probs_from_comp[depth_idx][seq_idx, prep_idx, :]
                actual = _dataset_probs(data.dataset, circuit, spinj.dim)
                self.assertTrue(np.allclose(actual, expected, atol=1e-10))

    def test_probabilities_from_compositions_unitary_noise(self):
        j = 0.5
        spinj = SpinJ(j)
        noise = jz_rotation(spinj, 0.3, 1.0)
        sim = SU2QuditRBSimulator(spinj, noise_channel=noise)
        design = SU2QuditRBDesign(j, depths=[2, 5], circuits_per_depth=3, seed=4)
        data = sim.run(design)

        compositions = sim.compute_nonspam_compositions(design)
        probs_from_comp = sim.probabilities_from_compositions(compositions)
        for depth_idx, circuits_at_depth in enumerate(design.circuit_lists):
            seq_at_depth = design.seq_index[depth_idx]
            prep_at_depth = design.prep_index[depth_idx]
            for circuit, seq_idx, prep_idx in zip(circuits_at_depth, seq_at_depth, prep_at_depth):
                expected = probs_from_comp[depth_idx][seq_idx, prep_idx, :]
                actual = _dataset_probs(data.dataset, circuit, spinj.dim)
                self.assertTrue(np.allclose(actual, expected, atol=1e-9))

    def test_depth_indices_subset(self):
        j = 0.5
        spinj = SpinJ(j)
        sim = SU2QuditRBSimulator(spinj)
        design = SU2QuditRBDesign(j, depths=[1, 2, 3], circuits_per_depth=2, seed=5)
        compositions = sim.compute_nonspam_compositions(design, depth_indices=[1])
        self.assertEqual(list(compositions.keys()), [1])
        self.assertEqual(compositions[1].shape, (2, spinj.dim ** 2, spinj.dim ** 2))

    def test_perturbed_spam_changes_probabilities(self):
        j = 0.5
        spinj = SpinJ(j)
        dim = spinj.dim
        sim = SU2QuditRBSimulator(spinj)
        design = SU2QuditRBDesign(j, depths=[3], circuits_per_depth=2, seed=6)
        compositions = sim.compute_nonspam_compositions(design)
        ideal_probs = sim.probabilities_from_compositions(compositions)

        # Perturb the state preps by a small unitary rotation and check the result
        # differs from the ideal-SPAM probabilities but each row still sums to 1.
        rng = np.random.default_rng(0)
        theta = 0.1
        H = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        H = H + H.conj().T
        import scipy.linalg as la
        U_pert = la.expm(-1j * theta * H)
        ideal_ops = sim._ideal_computational_ops()
        perturbed_statepreps = np.array([
            unitary_to_std_process_mx(U_pert) @ ideal_ops[ell] for ell in range(dim)
        ])
        perturbed_probs = sim.probabilities_from_compositions(compositions, statepreps=perturbed_statepreps)

        self.assertFalse(np.allclose(perturbed_probs[0], ideal_probs[0]))
        self.assertTrue(np.allclose(perturbed_probs[0].sum(axis=-1), 1.0, atol=1e-10))

    def test_non_hermitian_effect_raises(self):
        # `probabilities_from_compositions`'s `p = vdot(effect, rho)` formula relies on
        # the effect being Hermitian (so that Tr(E rho) is guaranteed real); a
        # non-Hermitian effect combined with a non-diagonal (but still physical,
        # Hermitian) prep should trip the imaginary-part guard.
        j = 0.5
        spinj = SpinJ(j)
        dim = spinj.dim
        sim = SU2QuditRBSimulator(spinj)
        design = SU2QuditRBDesign(j, depths=[1], circuits_per_depth=1, seed=7)
        compositions = sim.compute_nonspam_compositions(design)

        ideal_ops = sim._ideal_computational_ops()
        plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+><+|, Hermitian, non-diagonal
        statepreps = ideal_ops.copy()
        statepreps[0, :] = plus_state.ravel()

        non_hermitian_effect = np.zeros((dim, dim), dtype=complex)
        non_hermitian_effect[0, 1] = 1.0j  # E[0,1] = 1j != conj(E[1,0]) = 0 -- not Hermitian
        povm = ideal_ops.copy()
        povm[0, :] = non_hermitian_effect.ravel()

        with self.assertRaises(ValueError):
            sim.probabilities_from_compositions(compositions, statepreps=statepreps, povm=povm)


class TestSU2QuditRBSimulatorShots(BaseCase):

    def test_shots_totals_and_reproducibility(self):
        j = 1.5
        spinj = SpinJ(j)
        design = SU2QuditRBDesign(j, depths=[1, 2], circuits_per_depth=3, seed=7)

        sim_a = SU2QuditRBSimulator(spinj, shots=1000, seed=42)
        sim_b = SU2QuditRBSimulator(spinj, shots=1000, seed=42)
        data_a = sim_a.run(design)
        data_b = sim_b.run(design)

        for circuits_at_depth in design.circuit_lists:
            for circuit in circuits_at_depth:
                counts_a = data_a.dataset[circuit].counts
                counts_b = data_b.dataset[circuit].counts
                self.assertEqual(dict(counts_a), dict(counts_b))
                self.assertEqual(sum(counts_a.values()), 1000)
                for c in counts_a.values():
                    self.assertEqual(c, int(c))

    def test_different_seeds_differ(self):
        j = 1.5
        spinj = SpinJ(j)
        # Noiseless circuits have deterministic (one-hot) outcome distributions,
        # which would make multinomial sampling insensitive to the seed; use a noisy
        # simulator so the per-shot outcome distribution is nontrivial.
        noise = jz_dephasing(spinj, 0.3)
        design = SU2QuditRBDesign(j, depths=[3], circuits_per_depth=5, seed=8)
        sim_a = SU2QuditRBSimulator(spinj, noise_channel=noise, shots=200, seed=1)
        sim_b = SU2QuditRBSimulator(spinj, noise_channel=noise, shots=200, seed=2)
        data_a = sim_a.run(design)
        data_b = sim_b.run(design)
        any_different = any(
            dict(data_a.dataset[c].counts) != dict(data_b.dataset[c].counts)
            for c in design.circuit_lists[0]
        )
        self.assertTrue(any_different)


class TestSU2QuditRBSimulatorShortcut(BaseCase):

    def test_shortcut_matches_long_path_noiseless(self):
        j = 1.5
        spinj = SpinJ(j)
        sim = SU2QuditRBSimulator(spinj)
        design = SU2QuditRBDesign(j, depths=[1, 4, 6], circuits_per_depth=3, seed=9)
        for depth_idx in range(len(design.depths)):
            for angles in sim._unique_sequences_at_depth(design, depth_idx):
                S_shortcut = sim._compose(angles)
                S_full = sim._compose_full(angles)
                self.assertTrue(np.allclose(S_shortcut, S_full, atol=1e-10))


class TestSU2QuditRBSimulatorNoisePlacement(BaseCase):
    """Pins the noise-insertion order and skip-first-noise semantics of
    `_compose_full` against an independently hand-built expectation (rather than
    comparing `_compose`-derived quantities to each other, as the noiseless-shortcut
    tests above do)."""

    def test_two_gate_noise_placement_and_skip_semantics(self):
        spinj = SpinJ(j=1.5)
        N = jz_dephasing(spinj, 0.25, power=1.0)
        sim = SU2QuditRBSimulator(spinj, noise_channel=N)
        angles = np.array([
            [0.3, 0.5, 0.1],
            [1.2, 0.7, 2.1]
        ])
        U0, U1 = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
        S0 = unitary_to_std_process_mx(U0)
        S1 = unitary_to_std_process_mx(U1)
        # noise is skipped immediately after the first gate only; it is still applied
        # after the second (and every subsequent) gate.
        expected_skip = N @ S1 @ S0
        actual_skip = sim._compose_full(angles)
        self.assertTrue(np.allclose(actual_skip, expected_skip, atol=1e-12))
        return


class TestSU2QuditRBSimulatorNoiseFactory(BaseCase):
    """`noise_channel` also accepts a callable `(alpha, beta, gamma) -> channel`
    factory, called once per gate with that gate's own Euler angles, for
    gate-dependent errors."""

    def test_factory_matches_fixed_channel_when_gate_independent(self):
        j = 1.5
        spinj = SpinJ(j)
        N = jz_dephasing(spinj, 0.3, power=1.0)

        def factory(alpha, beta, gamma):
            return N

        sim_fixed = SU2QuditRBSimulator(spinj, noise_channel=N)
        sim_factory = SU2QuditRBSimulator(spinj, noise_channel=factory)
        self.assertFalse(sim_factory.is_noiseless)

        design = SU2QuditRBDesign(j, depths=[1, 3, 5], circuits_per_depth=3, seed=17)
        data_fixed = sim_fixed.run(design)
        data_factory = sim_factory.run(design)

        for circuits_at_depth in design.circuit_lists:
            for circuit in circuits_at_depth:
                probs_fixed = _dataset_probs(data_fixed.dataset, circuit, spinj.dim)
                probs_factory = _dataset_probs(data_factory.dataset, circuit, spinj.dim)
                self.assertTrue(np.allclose(probs_fixed, probs_factory, atol=1e-12))

    def test_gate_dependent_factory_matches_hand_computed_composition(self):
        spinj = SpinJ(j=1.5)

        def factory(alpha, beta, gamma):
            return jz_rotation(spinj, 0.05 * beta)

        sim = SU2QuditRBSimulator(spinj, noise_channel=factory)

        angles = np.array([
            [0.3, 0.5, 0.1],
            [1.2, 0.9, 2.1],
            [0.2, 0.6, 3.3]
        ])
        U0, U1, U2 = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
        S0 = unitary_to_std_process_mx(U0)
        S1 = unitary_to_std_process_mx(U1)
        S2 = unitary_to_std_process_mx(U2)
        N1 = unitary_to_std_process_mx(jz_rotation(spinj, 0.05 * angles[1, 1]))
        N2 = unitary_to_std_process_mx(jz_rotation(spinj, 0.05 * angles[2, 1]))

        # The two per-gate noise channels must differ, or this test would not
        # actually exercise gate-dependence.
        self.assertFalse(np.allclose(N1, N2, atol=1e-6))

        expected = N2 @ S2 @ N1 @ S1 @ S0
        actual = sim._compose_full(angles)
        self.assertTrue(np.allclose(actual, expected, atol=1e-12))

        # The gate-dependent result must differ from the noiseless composition and
        # from either single fixed-channel composition (i.e. applying one gate's
        # noise everywhere) -- confirming the factory is actually called per gate
        # rather than once globally.
        noiseless = S2 @ S1 @ S0
        fixed_n1_everywhere = N1 @ S2 @ N1 @ S1 @ S0
        fixed_n2_everywhere = N2 @ S2 @ N2 @ S1 @ S0
        self.assertFalse(np.allclose(actual, noiseless, atol=1e-6))
        self.assertFalse(np.allclose(actual, fixed_n1_everywhere, atol=1e-6))
        self.assertFalse(np.allclose(actual, fixed_n2_everywhere, atol=1e-6))


class TestNoiseChannelHelpers(BaseCase):

    def test_jz_dephasing_identity_at_zero_gamma(self):
        spinj = SpinJ(1.5)
        E = jz_dephasing(spinj, 0.0)
        self.assertEqual(E.shape, (spinj.dim ** 2, spinj.dim ** 2))
        self.assertTrue(np.allclose(E, np.eye(spinj.dim ** 2)))

    def test_jz_dephasing_negative_gamma_raises(self):
        spinj = SpinJ(0.5)
        with self.assertRaises(ValueError):
            jz_dephasing(spinj, -0.1)

    def test_jz_dephasing_decays_off_diagonal_coherences(self):
        spinj = SpinJ(1.5)
        dim = spinj.dim
        E = jz_dephasing(spinj, 1.0, power=1.0)
        rho = np.ones((dim, dim), dtype=complex)
        rho_vec = rho.ravel()
        evolved = (E @ rho_vec).reshape(dim, dim)
        for i in range(dim):
            for k in range(dim):
                self.assertAlmostEqual(evolved[i, k].real, np.exp(-abs(i - k)))

    def test_jz_rotation_identity_at_zero_theta(self):
        spinj = SpinJ(1.5)
        U = jz_rotation(spinj, 0.0)
        self.assertEqual(U.shape, (spinj.dim, spinj.dim))
        self.assertTrue(np.allclose(U, np.eye(spinj.dim)))

    def test_jz_rotation_is_unitary_and_diagonal_in_spins(self):
        spinj = SpinJ(1.5)
        U = jz_rotation(spinj, 0.7, power=2.0)
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(spinj.dim)))
        expected_diag = np.exp(1j * 0.7 * spinj.spins ** 2.0)
        self.assertTrue(np.allclose(np.diag(U), expected_diag))

    def test_compose_noise_channels_order_and_shapes(self):
        spinj = SpinJ(0.5)
        dim = spinj.dim
        U = jz_rotation(spinj, 0.4)
        S = jz_dephasing(spinj, 0.1)
        composed = compose_noise_channels(spinj, U, S)
        expected = S @ unitary_to_std_process_mx(U)
        self.assertEqual(composed.shape, (dim ** 2, dim ** 2))
        self.assertTrue(np.allclose(composed, expected))

    def test_compose_noise_channels_bad_shape_raises(self):
        spinj = SpinJ(0.5)
        with self.assertRaises(ValueError):
            compose_noise_channels(spinj, np.eye(3))


class TestSU2QuditRBSimulatorConstruction(BaseCase):

    def test_accepts_spin_value_or_spinj(self):
        sim_from_j = SU2QuditRBSimulator(1.5)
        sim_from_spinj = SU2QuditRBSimulator(SpinJ(1.5))
        self.assertEqual(sim_from_j.dim, sim_from_spinj.dim)
        self.assertTrue(sim_from_j.is_noiseless)

    def test_bad_noise_channel_shape_raises(self):
        with self.assertRaises(ValueError):
            SU2QuditRBSimulator(0.5, noise_channel=np.eye(3))

    def test_dim_mismatch_with_design_raises(self):
        sim = SU2QuditRBSimulator(0.5)
        design = SU2QuditRBDesign(1.5, depths=[1], circuits_per_depth=2, seed=10)
        with self.assertRaises(ValueError):
            sim.run(design)


# ---------------------------------------------------------------------------
# predicted_zero_noise_variance
# ---------------------------------------------------------------------------

class TestPredictedZeroNoiseVariance(BaseCase):
    """
    Golden-table tests against the R1RB columns of the paper's Tables
    `variancewithk` and `variancewithj` (6-significant-figure values quoted from the
    paper's LaTeX source), plus the input-validation contract of
    `predicted_zero_noise_variance(j, k)`. `predicted_zero_noise_variance` only
    implements the R1RB formula, so there are no other variant columns to test.
    """

    def _assert_close_6sf(self, actual, golden):
        # The golden literals themselves are only quoted to 6 significant figures, so
        # allow a couple of ULPs of rounding slop on top of that (verified against an
        # independent from-scratch double-precision recomputation of both paper
        # tables).
        self.assertAlmostEqual(actual, golden, delta=abs(golden) * 2e-5 + 1e-9)

    def test_variancewithk_table_j_seven_halves(self):
        # Table `variancewithk`: j = 7/2, k = 0..7, R1RB column.
        j = 3.5
        golden_ssr1rb = {
            0: 0.0,
            1: 0.269048,
            2: 0.540816,
            3: 0.773292,
            4: 1.02387,
            5: 1.28994,
            6: 1.62223,
            7: 2.11888,
        }
        for k, ssr1 in golden_ssr1rb.items():
            self._assert_close_6sf(predicted_zero_noise_variance(j, k), ssr1)

    def test_variancewithj_table(self):
        # Table `variancewithj`: k = 2j fixed, j = 0..7/2, R1RB column.
        golden_ssr1rb = {
            0.0: 0,
            0.5: 1,
            1.0: 1.40476,
            1.5: 1.63867,
            2.0: 1.80578,
            2.5: 1.9322,
            3.0: 2.03407,
            3.5: 2.11888,
        }
        for j, ssr1 in golden_ssr1rb.items():
            k = round(2 * j)
            if k == 0:
                self.assertEqual(predicted_zero_noise_variance(j, k), 0.0)
                self.assertEqual(ssr1, 0)
                continue
            self._assert_close_6sf(predicted_zero_noise_variance(j, k), ssr1)

    def test_bad_k_raises(self):
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 4)  # 2j == 3 < 4
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, -1)
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 0.5)  # not an integer


# ---------------------------------------------------------------------------
# SU2QuditRB / SU2QuditRBResults
# ---------------------------------------------------------------------------

def _dataset_from_probs(edesign, probs_by_depth, dim):
    """
    Build a `ProtocolData` (exact-probability `DataSet`, following
    `SU2QuditRBSimulator.run`'s `shots=None` convention) directly from precomputed
    per-(depth, sequence, prep) outcome probabilities -- e.g. from
    `SU2QuditRBSimulator.probabilities_from_compositions` -- rather than from a live
    `SU2QuditRBSimulator.run` call. Used by the SPAM-robustness test below to feed
    perturbed-SPAM probabilities through `SU2QuditRB` without
    re-simulating circuits.

    Parameters
    ----------
    edesign : SU2QuditRBDesign

    probs_by_depth : dict
        Depth index -> array of shape `(circuits_per_depth, dim, dim)`, as returned
        by `probabilities_from_compositions` (indexed `[seq_index, prep_index, :]`).

    dim : int

    Returns
    -------
    ProtocolData
    """
    ds = DataSet(collision_action="aggregate")
    ds.repType = np.float64
    outcome_labels = [str(ell) for ell in range(dim)]
    for depth_idx, circuits_at_depth in enumerate(edesign.circuit_lists):
        seq_at_depth = edesign.seq_index[depth_idx]
        prep_at_depth = edesign.prep_index[depth_idx]
        probs_arr = probs_by_depth[depth_idx]
        for circuit, seq_idx, prep_idx in zip(circuits_at_depth, seq_at_depth, prep_at_depth):
            probs = probs_arr[seq_idx, prep_idx, :]
            counts = {ol: float(p) for ol, p in zip(outcome_labels, probs)}
            ds.add_count_dict(circuit, counts)
    ds.done_adding_data()
    return ProtocolData(edesign, ds)


class TestSU2QuditRBRequiresCharcores(BaseCase):
    """`SU2QuditRB` needs the `charcores` aux data that every `SU2QuditRBDesign`
    provides; an edesign-like object without it should fail loudly (`TypeError`),
    not silently misbehave. (There is no longer a "plain" design variant lacking
    `charcores` to use as a foil, so this drives `_per_sequence_irrep_values`
    directly against a minimal stand-in, the way
    `TestSU2QuditRBWeightNormalization` below does.)"""

    def test_rejects_edesign_without_charcores(self):
        class _NoCharcoresEdesign:
            dim = 2
            circuits_per_depth = 1
            circuit_lists = [[]]
            seq_index = [[]]
            prep_index = [[]]

        with self.assertRaises(TypeError):
            SU2QuditRB()._per_sequence_irrep_values(_NoCharcoresEdesign(), {}, 0, np.eye(2))


class _FakeCircuitRow:
    """Stand-in for a `DataSet` row: `_reconstruct_prep_effect_probs` only ever reads
    `.fractions` off of what `ds[circuit]` returns."""

    __slots__ = ('fractions',)

    def __init__(self, fractions):
        self.fractions = fractions


class _FakeSU2RBEdesign:
    """
    Minimal stand-in for a single-depth `SU2QuditRBDesign`, exposing only the attributes
    `SU2QuditRB._per_sequence_irrep_values` actually reads (`dim`,
    `circuits_per_depth`, `circuit_lists`, `seq_index`, `prep_index`, `charcores`).
    Used to drive that method directly on hand-fabricated probability/weight data,
    without going through a real `SU2QuditRBDesign`/`SU2QuditRBSimulator`/`DataSet`.
    """

    def __init__(self, dim, circuits_per_depth, circuits, seq_index, prep_index, charcores_rows):
        self.dim = dim
        self.circuits_per_depth = circuits_per_depth
        self.circuit_lists = [circuits]
        self.seq_index = [seq_index]
        self.prep_index = [prep_index]
        self.charcores = [charcores_rows]


def _branch_character_transform_single_length(probs_branch, weights, irrep_sizes):
    """
    Literal pure-numpy transcription of `SU2CharacterRBSim.character_transform`
    from the `su2-rb-conservative` branch (`su2rbsims.py:582-615`), specialized to a
    single RB length (so the branch's `num_lens` axis -- irrelevant to the weight
    normalization being locked down here -- collapses out).

    Parameters
    ----------
    probs_branch : numpy array
        Shape `(num_statepreps, circuits_per_length, num_effects)` -- the branch's
        `_probs[:, j, :, :]` for a single length `j`.

    weights : numpy array
        Shape `(circuits_per_length, num_irreps)` -- the branch's `_chars[j, :, :]`
        (here standing in for either `characters` or, as used below, `charcores`).

    irrep_sizes : numpy array
        Shape `(num_irreps,)`.

    Returns
    -------
    numpy array
        Shape `(num_statepreps, num_effects, num_irreps)` -- the branch's
        `arr[:, :, :, j]`.
    """
    circuits_per_length = probs_branch.shape[1]
    wchars = weights * irrep_sizes[np.newaxis, :]                # branch: wchars = _chars * irrep_sizes[...]
    permprobs = np.moveaxis(probs_branch, 1, 2)                   # branch: np.moveaxis(probs, (1,2,3), (2,3,1))
    # branch: P = np.tensordot(currprobs, currchars, axes=1) / circuits_per_length
    return np.tensordot(permprobs, wchars, axes=1) / circuits_per_length


def _branch_synspam_character_transform_single_length(probs_branch, weights, irrep_sizes, M):
    """
    Literal pure-numpy transcription of `SU2CharacterRBSim.
    synspam_character_transform` (`su2rbsims.py:617-657`), specialized to a single
    RB length (see `_branch_character_transform_single_length`).

    Returns
    -------
    numpy array
        Shape `(num_irreps,)` -- the branch's `synthetic_probs[:, j]`.
    """
    block_full = _branch_character_transform_single_length(probs_branch, weights, irrep_sizes)
    num_irreps = block_full.shape[2]
    synthetic_probs = np.zeros(num_irreps)
    for k in range(num_irreps):
        block = block_full[:, :, k]           # branch: block = P[:,:,k]
        row_M_k = M[k, :]                      # branch: row_M_k = M[k,:]
        synthetic_probs[k] = row_M_k @ block @ row_M_k
    return synthetic_probs


class TestSU2QuditRBWeightNormalization(BaseCase):
    """
    Locks the (2k+1)-and-charcore weight normalization scale of
    `SU2QuditRB._per_sequence_irrep_values` to the `su2-rb-conservative`
    branch's normative `character_transform`/`synspam_character_transform` (see
    `SU2QuditRB`'s docstring). Only the rank-1 half is tested here, since
    `SU2QuditRB` is the only surviving protocol -- there is no
    character-weighted variant left to test.

    Unlike the fit-based end-to-end tests elsewhere in this module -- whose
    `A_k * f_k**x` fit absorbs any constant per-irrep scale error into `A_k` without
    affecting `f_k`, and whose 0.1-10x variance-ratio sanity check tolerates over a
    3x scale error -- this test compares raw, pre-fit, pre-average per-sequence
    values directly against a literal transcription of the branch's tensor
    contraction (`_branch_synspam_character_transform_single_length`), on
    hand-fabricated (not necessarily physical) probability and weight arrays, at
    1e-12. A constant per-irrep scale error in the weighting factors would be caught
    here even though it passes every other test in this module.
    """

    def test_rank1_weighting_matches_branch(self):
        rng = np.random.default_rng(2024)
        dim = 3
        circuits_per_depth = 5

        # Random (valid) prep-by-effect probability matrices, one per sampled
        # sequence, and random per-sequence irrep weights standing in for
        # `charcores` (the weighting arithmetic being locked down here doesn't care
        # whether these are honest P_k values).
        raw = rng.uniform(0.1, 1.0, size=(circuits_per_depth, dim, dim))
        P = raw / raw.sum(axis=2, keepdims=True)  # P[s, l, :] is a valid distribution
        weights = rng.uniform(-2.0, 2.0, size=(circuits_per_depth, dim))
        M = rng.normal(size=(dim, dim))  # need not be an actual SpinJ M for this algebraic check
        irrep_sizes = 2 * np.arange(dim) + 1

        # --- Branch reference, from the literal transcription above. ---
        probs_branch = np.moveaxis(P, 0, 1)  # (s, l, m) -> (l=prep, s=circuit, m=effect)
        reference = _branch_synspam_character_transform_single_length(probs_branch, weights, irrep_sizes, M)

        # --- The actual production code path: build a minimal fake edesign/ds ---
        # --- exposing exactly what `_per_sequence_irrep_values` reads, and call ---
        # --- it directly, then average over sequences (matching the branch's ---
        # --- `/circuits_per_length` normalization, which is baked into ---
        # --- `_branch_character_transform_single_length` above). ---
        circuits = [(s, l) for s in range(circuits_per_depth) for l in range(dim)]
        seq_index = [s for s in range(circuits_per_depth) for l in range(dim)]
        prep_index = [l for s in range(circuits_per_depth) for l in range(dim)]
        charcores_rows = [weights[s, :].tolist() for s in seq_index]
        ds = {(s, l): _FakeCircuitRow({(str(m),): P[s, l, m] for m in range(dim)})
              for s in range(circuits_per_depth) for l in range(dim)}
        edesign = _FakeSU2RBEdesign(dim, circuits_per_depth, circuits, seq_index, prep_index, charcores_rows)

        X = SU2QuditRB()._per_sequence_irrep_values(edesign, ds, 0, M)
        family_mean = X.mean(axis=0)

        self.assertTrue(np.allclose(family_mean, reference, atol=1e-12, rtol=0),
                         msg=f"family_mean={family_mean} reference={reference}")


class TestSU2QuditRBNoiseless(BaseCase):
    """Noiseless analytic end-to-end check: all f_k should be (statistically)
    consistent with 1, and the empirical per-sequence-estimator variance should be
    the right order of magnitude relative to `predicted_zero_noise_variance`. Unlike
    an unweighted, un-hidden-gate SSRB estimator (removed under the R1RB-only
    strip-down), R1RB has nonzero zero-noise variance, so exact agreement isn't
    expected at finite circuit counts -- this is a statistical check."""

    def test_rank1_rb_noiseless(self):
        j = 1.5
        spinj = SpinJ(j)
        dim = spinj.dim
        sim = SU2QuditRBSimulator(spinj)
        design = SU2QuditRBDesign(j, depths=[1, 2, 3, 4, 5], circuits_per_depth=300, seed=21)
        data = sim.run(design)
        results = SU2QuditRB().run(data)

        # f_k consistent with 1 within a generous multiple of its own stderr.
        z = np.abs(results.decays - 1.0) / (results.decay_stderrs + 1e-300)
        self.assertTrue(np.all(z[1:] < 6.0), msg=f"z-scores vs 1.0: {z}")

        # Order-of-magnitude cross-check against the paper's zero-noise variance
        # formula (predicted_zero_noise_variance), evaluated at the shortest depth.
        diag = results.variance_diagnostic(depth_index=0)
        for k in range(1, dim):
            predicted, empirical = diag[k]
            self.assertGreater(predicted, 0.0)
            self.assertGreater(empirical, 0.0)
            ratio = empirical / predicted
            self.assertTrue(0.1 < ratio < 10.0,
                             msg=f"k={k} predicted={predicted} empirical={empirical} ratio={ratio}")

        # Pin the *absolute* paper normalization directly on the raw (pre-fit)
        # per-depth sample means, not just the fitted decays: the A_k*f_k^x fit
        # above is scale-blind (a constant per-irrep scale error would just be
        # absorbed into A_k and leave f_k unaffected), so this is the test that
        # actually locks `per_irrep_means` itself to 1 within its own stderr, at
        # every depth, for every nontrivial irrep.
        means = results.per_irrep_means
        stderrs = results.per_irrep_stderrs
        z_by_depth = np.abs(means[1:, :] - 1.0) / stderrs[1:, :]
        self.assertTrue(np.all(z_by_depth < 6.0),
                         msg=f"per-depth per_irrep_means z-scores vs 1.0: {z_by_depth}")


class TestSU2QuditRBMatchesGateNoise(BaseCase):
    """Noisy analytic end-to-end check: with a known gate-independent noise channel,
    R1RB should recover the same per-irrep fidelities as the closed-form
    `f_k = Tr(P_k . Lambda) / (2k+1)` twirl formula -- gate noise is what R1RB
    measures; only its SPAM-robustness properties are special (see
    `TestSU2QuditRBSpamRobustness`)."""

    def test_rank1_rb_matches_gate_noise(self):
        j = 1.5
        gamma = 0.05
        n_circuits = 300
        max_depth = 15
        spinj = SpinJ(j)
        dim = spinj.dim
        noise = jz_dephasing(spinj, gamma, power=1.0)
        sim = SU2QuditRBSimulator(spinj, noise_channel=noise, shots=None, seed=1)
        depths = list(range(1, max_depth + 1))
        design = SU2QuditRBDesign(j, depths, circuits_per_depth=n_circuits, seed=22)
        data = sim.run(design)
        results = SU2QuditRB().run(data)

        analytic_f = np.array([
            np.trace(spinj.irrep_stdmx_projectors[k] @ noise).real / (2 * k + 1)
            for k in range(dim)
        ])
        tol = 5.0 * results.decay_stderrs + 1e-3
        self.assertTrue(np.all(np.abs(results.decays - analytic_f) <= tol),
                         msg=f"decays={results.decays} analytic={analytic_f} tol={tol}")
        self.assertAlmostEqual(results.decays[0], 1.0, delta=1e-6)


class TestSU2QuditRBApi(BaseCase):
    """Protocol/Results API-surface checks: DataFrame export, isinstance, and the
    write()/read-from-dir serialization round trip -- retargeted from the former
    (plain-SSRB) Phase 5 tests at the single surviving protocol/results pair."""

    def test_rates_dataframe_and_variance_diagnostic(self):
        j = 0.5
        spinj = SpinJ(j)
        sim = SU2QuditRBSimulator(spinj)
        # >=3 depths: with exactly 2, the 2-parameter (a, f) fit has zero residual
        # degrees of freedom, so scipy.optimize.curve_fit can't estimate a covariance
        # and emits an OptimizeWarning (which pytest.ini's filterwarnings policy
        # turns into an error by default) -- see test_failed_irrep_fit_warns, which
        # exercises that scenario deliberately.
        design = SU2QuditRBDesign(j, depths=[1, 2, 3], circuits_per_depth=3, seed=1)
        data = sim.run(design)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            results = SU2QuditRB().run(data)

        df = results.rates_dataframe()
        self.assertEqual(len(df), spinj.dim)
        self.assertEqual(list(df['irrep']), list(range(spinj.dim)))
        for col in ('decay_f', 'decay_f_stderr', 'rate_p', 'rate_p_stderr'):
            self.assertIn(col, df.columns)

        # predicted_zero_noise_variance(j, k) is the R1RB rank-1 formula -- unlike
        # the removed plain-SSRB variant, it is not identically 0 for k > 0.
        diag = results.variance_diagnostic()
        self.assertEqual(set(diag.keys()), set(range(spinj.dim)))
        for k, (predicted, empirical) in diag.items():
            self.assertGreaterEqual(predicted, 0.0)
            self.assertGreaterEqual(empirical, 0.0)

    def test_isinstance_results(self):
        j = 0.5
        # >=3 depths: see test_rates_dataframe_and_variance_diagnostic's comment.
        design = SU2QuditRBDesign(j, depths=[1, 2, 3], circuits_per_depth=2, seed=0)
        data = SU2QuditRBSimulator(j).run(design)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            results = SU2QuditRB().run(data)
        self.assertIsInstance(results, SU2QuditRBResults)

    def test_serialization_round_trip(self):
        # Regression test: SU2QuditRBResults registered no auxfile_types for its
        # ndarray/FitResults-list attributes, so write() raised ValueError (plain JSON
        # serialization can't handle numpy arrays or FitResults objects).
        j = 0.5
        design = SU2QuditRBDesign(j, depths=[1, 2, 3], circuits_per_depth=2, seed=0)
        data = SU2QuditRBSimulator(j).run(design)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            results = SU2QuditRB().run(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            results.write(tmpdir)
            from pygsti.io import read_results_from_dir
            reloaded_dir = read_results_from_dir(tmpdir)
        reloaded = reloaded_dir.for_protocol[results.name]

        self.assertIsInstance(reloaded, SU2QuditRBResults)
        self.assertEqual(reloaded.j, results.j)
        self.assertEqual(reloaded.dim, results.dim)
        self.assertEqual(reloaded.depths, results.depths)
        self.assertTrue(np.array_equal(reloaded.per_irrep_means, results.per_irrep_means))
        self.assertTrue(np.array_equal(reloaded.per_irrep_stderrs, results.per_irrep_stderrs))
        self.assertTrue(np.array_equal(reloaded.per_irrep_variances, results.per_irrep_variances))
        self.assertTrue(np.array_equal(reloaded.decays, results.decays))
        self.assertTrue(np.array_equal(reloaded.decay_stderrs, results.decay_stderrs))
        self.assertTrue(np.array_equal(reloaded.rates, results.rates))
        self.assertTrue(np.array_equal(reloaded.rates_covariance, results.rates_covariance))
        self.assertEqual(len(reloaded.fits), len(results.fits))
        for f_orig, f_reloaded in zip(results.fits, reloaded.fits):
            self.assertEqual(f_orig.estimates, f_reloaded.estimates)

    def test_failed_irrep_fit_warns(self):
        # A single-depth design under-determines the 2-parameter (a, f) fit for every
        # irrep: scipy.optimize.curve_fit still finds a parameter estimate from the
        # single data point, but can't estimate its covariance, and warns
        # (OptimizeWarning) rather than raising. This is therefore an honest inf
        # standard error, not a fit failure: success stays True and decays/rates
        # remain finite; only their uncertainty is unknown. See
        # test_genuine_irrep_fit_failure_warns_rbfitfailurewarning below for the
        # (distinct) genuine-failure path this used to be conflated with.
        j = 0.5
        design = SU2QuditRBDesign(j, depths=[1], circuits_per_depth=2, seed=0)
        data = SU2QuditRBSimulator(j).run(design)
        with self.assertWarns(OptimizeWarning):
            results = SU2QuditRB().run(data)
        self.assertTrue(np.all(np.isinf(results.decay_stderrs)))
        self.assertTrue(np.all(np.isfinite(results.decays)))
        self.assertTrue(np.all(np.isfinite(results.rates)))
        self.assertTrue(all(fit.success for fit in results.fits))

    def test_genuine_irrep_fit_failure_warns_rbfitfailurewarning(self):
        # A genuine per-irrep fit failure (as opposed to the merely-unestimable-
        # covariance case above) should surface as the pyGSTi-native
        # RBFitFailureWarning, and should poison every entry of
        # `rates`/`rates_covariance` with nan -- not just the failed irrep's -- since
        # F mixes all irreps together. Drives `_fit_and_get_rates` directly with a
        # hand-crafted series (irrep 1 is all-nan, which scipy.optimize.curve_fit
        # rejects outright with a ValueError) so the genuine-failure branch of
        # `_fit_irrep_decay` is exercised deterministically.
        j = 0.5
        spinj = SpinJ(j)
        F = spinj.decay_recoupling_matrix
        x = np.array([2.0, 3.0, 4.0])
        per_irrep_series = np.array([
            [1.0, 1.0, 1.0],
            [np.nan, np.nan, np.nan],
        ])
        with self.assertWarns(RBFitFailureWarning) as cm:
            fits, f, sigma_f, p, cov_p = SU2QuditRB()._fit_and_get_rates(x, per_irrep_series, F)
        self.assertIn('irrep', str(cm.warning))
        self.assertEqual([fit.success for fit in fits], [True, False])
        self.assertTrue(np.all(np.isnan(p)))
        self.assertTrue(np.all(np.isnan(cov_p)))


class TestSU2QuditRBSpamRobustness(BaseCase):
    """
    The paper's central SPAM-robustness signature: `SU2QuditRB` stays
    unbiased under a SPAM perturbation that would bias an unweighted estimator.

    Every `SU2QuditRBDesign` is hidden-first-gate (there is no net-identity, "plain
    SSRB", `invert_from=0` circuit convention to fall back on), so its unweighted
    `diag(M P M^T)[k]` sandwich already averages to (statistically) zero
    for k > 0 *regardless of SPAM correctness*: a Haar-random hidden first gate
    integrates any unweighted, nontrivial-irrep contribution to zero by Legendre
    orthogonality, with or without a SPAM perturbation. (Verified numerically while
    developing this test: even a large, non-Jz-diagonal SPAM perturbation and tens
    of thousands of sampled sequences produced no statistically resolvable shift in
    the unweighted sandwich's mean on hidden-gate-design data -- the signal the
    rank-1/charcore weighting exists to recover is exactly what a naive unweighted
    average discards.) So the unweighted foil has to be computed on its own native
    circuit convention -- net ideal composition equal to the identity -- to show
    anything at all, let alone a SPAM-induced bias.

    That circuit family no longer has a production class of its own, so it is
    reconstructed here directly from public `su2tools` primitives
    (`random_euler_angles`, `composition_inverse`, `SpinJ.unitaries_from_angles`)
    plus `unitary_to_std_process_mx` -- a few lines of numpy, no production
    `su2rb.py` code touched, no deleted class revived. On these net-identity
    sequences, the unweighted `diag(M P M^T)[k]` sandwich matches the true
    (analytic) per-irrep gate-noise fidelity `f_k**m` under ideal SPAM but acquires
    a large bias under a fixed, non-Jz-diagonal SPAM perturbation -- the classic
    "plain SSRB is not SPAM-robust" signature. `SU2QuditRB`, run on the
    standard (hidden-gate) `SU2QuditRBDesign` pipeline under the *same* perturbation,
    stays statistically consistent with the true fidelity.

    Seeded throughout; j=1.5 keeps the (already fast) 4x4-superoperator composition
    cheap enough to run in a few seconds.
    """

    @staticmethod
    def _sample_net_identity_angles(m, rng):
        """
        One `(m+1, 3)` Euler-angle sequence whose net ideal composition (all `m+1`
        rows) is the identity: rows `0..m-1` are Haar-random, row `m` is the exact
        composition-inverse of rows `0..m-1`. This is the old, pre-strip-down
        "plain SSRB" (`invert_from=0`) circuit convention -- unlike every surviving
        `SU2QuditRBDesign`, there is no "hidden" gate here for the inverting row to
        leave unaccounted for.
        """
        angles = np.zeros((m + 1, 3))
        a, b, g = random_euler_angles(m, rng=rng)
        angles[:m, 0], angles[:m, 1], angles[:m, 2] = a, b, g
        angles[m, :] = composition_inverse(angles[:m, 0], angles[:m, 1], angles[:m, 2])
        return angles

    def _net_identity_unweighted_mean(self, spinj, noise_superop, M, m, n_circuits, rng, statepreps, povm):
        """
        The unweighted `diag(M P M^T)[k]` sandwich (no charcore weighting, no
        irrep-size scaling -- "what you'd get without the rank-1 weighting"),
        averaged over `n_circuits` freshly-sampled net-identity sequences
        (`_sample_net_identity_angles`) at depth `m`, with `noise_superop` applied
        after *every* gate (there is no hidden gate to skip noise after in this
        convention).

        Returns
        -------
        mean, stderr : numpy array, shape (dim,)
        """
        dim = spinj.dim
        vals = np.zeros((n_circuits, dim))
        for s in range(n_circuits):
            angles = self._sample_net_identity_angles(m, rng)
            Us = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
            S = np.eye(dim * dim, dtype=complex)
            for U in Us:
                S = noise_superop @ (unitary_to_std_process_mx(U) @ S)
            evolved = np.einsum('ab,ib->ia', S, statepreps)
            P = np.real(np.einsum('ek,ik->ie', np.conj(povm), evolved))
            vals[s, :] = np.einsum('kl,lm,km->k', M, P, M)
        mean = vals.mean(axis=0)
        stderr = vals.std(axis=0, ddof=1) / np.sqrt(n_circuits)
        return mean, stderr

    def test_unweighted_foil_biased_rank1_unbiased(self):
        import scipy.linalg as _sla

        j = 1.5
        gamma = 0.05
        n_circuits = 200
        max_depth = 10
        spinj = SpinJ(j)
        dim = spinj.dim
        M = spinj.synthetic_spam_matrix

        noise = jz_dephasing(spinj, gamma, power=1.0)
        sim = SU2QuditRBSimulator(spinj, noise_channel=noise)

        # A fixed, non-Jz-diagonal (Hermitian-generator) unitary "miscalibration" of
        # the state preps and POVM effects, applied identically across every circuit.
        rng = np.random.default_rng(7)
        theta_prep, theta_povm = 0.3, 0.25
        H_prep = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        H_prep = H_prep + H_prep.conj().T
        H_povm = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        H_povm = H_povm + H_povm.conj().T
        U_prep = _sla.expm(-1j * theta_prep * H_prep)
        U_povm = _sla.expm(-1j * theta_povm * H_povm)

        ideal_ops = sim._ideal_computational_ops()
        perturbed_preps = np.array([
            unitary_to_std_process_mx(U_prep) @ ideal_ops[ell] for ell in range(dim)])
        perturbed_povm = np.array([
            unitary_to_std_process_mx(U_povm) @ ideal_ops[ell] for ell in range(dim)])

        analytic_f = np.array([
            np.trace(spinj.irrep_stdmx_projectors[k] @ noise).real / (2 * k + 1)
            for k in range(dim)
        ])

        # --- SU2QuditRB (Legendre/charcore-weighted), on the standard
        # --- hidden-gate SU2QuditRBDesign pipeline: stays unbiased under the SPAM
        # --- perturbation. ---
        depths = list(range(1, max_depth + 1))
        design = SU2QuditRBDesign(j, depths, circuits_per_depth=n_circuits, seed=11)
        compositions = sim.compute_nonspam_compositions(design)
        perturbed_probs = sim.probabilities_from_compositions(
            compositions, statepreps=perturbed_preps, povm=perturbed_povm)
        data_perturbed = _dataset_from_probs(design, perturbed_probs, dim)
        results_r1 = SU2QuditRB().run(data_perturbed)

        z_r1 = np.abs(results_r1.decays - analytic_f) / results_r1.decay_stderrs
        self.assertTrue(np.all(z_r1[1:] < 6.0), msg=f"R1RB z-scores vs analytic: {z_r1}")

        # --- Unweighted foil, on freshly-sampled net-identity sequences (see this
        # --- class's docstring for why the hidden-gate design's own data can't be
        # --- reused here): matches the analytic fidelity f_k**m under ideal SPAM,
        # --- but acquires a large bias under the same SPAM perturbation. Standard
        # --- RB theory's exponent here is the number of *random* gates `m`, not
        # --- `m + 1`: the (deterministic) inverting gate's own noise contributes
        # --- only to the amplitude, not the decay exponent.
        m = max_depth
        rng_foil = np.random.default_rng(23)
        noise_superop = sim._noise_superop
        mean_ideal, se_ideal = self._net_identity_unweighted_mean(
            spinj, noise_superop, M, m, n_circuits, rng_foil, ideal_ops, ideal_ops)
        mean_perturbed, se_perturbed = self._net_identity_unweighted_mean(
            spinj, noise_superop, M, m, n_circuits, rng_foil, perturbed_preps, perturbed_povm)

        target = analytic_f ** m
        z_unweighted_ideal = np.abs(mean_ideal - target) / se_ideal
        self.assertTrue(np.all(z_unweighted_ideal[1:] < 6.0),
                         msg=f"unweighted foil (ideal SPAM) z-scores vs analytic: {z_unweighted_ideal}")
        z_unweighted_perturbed = np.abs(mean_perturbed - target) / se_perturbed
        self.assertTrue(np.all(z_unweighted_perturbed[1:] > 8.0),
                         msg=f"unweighted foil (perturbed SPAM) z-scores vs analytic: {z_unweighted_perturbed}")
