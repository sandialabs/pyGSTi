"""
Tests for pygsti.protocols.su2rb (Phase 3: SU(2) synthetic SPAM RB circuits and
experiment designs; Phase 4: the SU2RBDataSimulator).
"""
import tempfile
import warnings

import numpy as np

from pygsti.circuits.circuit import Circuit
from pygsti.protocols.su2rb import (
    SU2RBDesign,
    SU2CharacterRBDesign,
    SU2RBDataSimulator,
    SyntheticSPAMRB,
    SyntheticSPAMRBResults,
    predicted_zero_noise_variance,
    circuit_from_euler_angles,
    euler_angles_from_circuit,
    jz_dephasing,
    jz_rotation,
    compose_noise_channels,
    GATE_NAME,
    POVM_NAME,
)
from pygsti.tools.su2tools import SpinJ, distance_mod_phase
from pygsti.tools.optools import unitary_to_std_process_mx

from ..util import BaseCase


# ---------------------------------------------------------------------------
# circuit_from_euler_angles / euler_angles_from_circuit
# ---------------------------------------------------------------------------

class TestEulerAngleCircuitHelpers(BaseCase):

    def test_round_trip_no_prep(self):
        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 4 * np.pi, size=(5, 3))
        c = circuit_from_euler_angles(angles, j=1.5, qudit_label='Q0')
        self.assertEqual(len(c), 5)
        for layer in c:
            self.assertEqual(layer.name, GATE_NAME)

        recovered = euler_angles_from_circuit(c)
        # The plan asks for agreement to <= 1e-12; empirically (see the Task 3.0
        # spike notes) the string round trip is bit-exact, so we check that too.
        self.assertTrue(np.allclose(recovered, angles, atol=1e-12))
        self.assertTrue(np.array_equal(recovered, angles))

    def test_round_trip_through_string(self):
        rng = np.random.default_rng(1)
        angles = rng.uniform(0, 4 * np.pi, size=(4, 3))
        c = circuit_from_euler_angles(angles, j=0.5, qudit_label='Q0', prep_index=1)
        s = c.str
        c2 = Circuit(None, stringrep=s, line_labels=c.line_labels)
        self.assertEqual(c, c2)
        recovered = euler_angles_from_circuit(c2)
        self.assertTrue(np.array_equal(recovered, angles))

    def test_prep_and_povm_labels_embedded(self):
        angles = np.zeros((2, 3))
        c = circuit_from_euler_angles(angles, j=1.5, qudit_label='Q0', prep_index=2)
        names = [layer.name for layer in c]
        self.assertEqual(names[0], 'rho2')
        self.assertEqual(names[-1], 'Mdefault')
        self.assertEqual(names.count(GATE_NAME), 2)

    def test_prep_index_out_of_range_raises(self):
        angles = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            circuit_from_euler_angles(angles, j=0.5, qudit_label='Q0', prep_index=2)  # dim=2, valid range 0,1

    def test_bad_angle_shape_raises(self):
        with self.assertRaises(ValueError):
            circuit_from_euler_angles(np.zeros((3, 2)), j=0.5)

    def test_euler_angles_from_circuit_requires_gu_layer(self):
        c = Circuit(['rho0', 'Mdefault'], line_labels=('Q0',))
        with self.assertRaises(ValueError):
            euler_angles_from_circuit(c)


# ---------------------------------------------------------------------------
# SU2RBDesign / SU2CharacterRBDesign
# ---------------------------------------------------------------------------

def _net_unitary(spinj, angles):
    Us = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
    net = np.eye(spinj.dim, dtype=complex)
    for U in Us:
        net = U @ net
    return net


class _SU2RBDesignChecks:
    """Mixin providing the shared design-level tests, parameterized by spin `j`."""

    j = None
    design_cls = SU2RBDesign

    def test_circuit_counts(self):
        depths = [1, 2, 4]
        circuits_per_depth = 3
        design = self.design_cls(self.j, depths, circuits_per_depth, seed=1234)
        dim = design.dim
        self.assertEqual(dim, round(2 * self.j) + 1)
        for circuits_at_depth in design.circuit_lists:
            self.assertEqual(len(circuits_at_depth), circuits_per_depth * dim)

    def _extra_paired_attrs(self):
        """Names of additional `paired_with_circuit_attrs`-style aux lists that
        subclasses (e.g. `SU2CharacterRBDesign`'s `characters`/`charcores`) add on top
        of the base `euler_angles`/`seq_index`/`prep_index` set. Overridden by the
        character design test classes below."""
        return []

    def test_aux_lists_aligned_with_circuits(self):
        depths = [1, 3]
        circuits_per_depth = 2
        design = self.design_cls(self.j, depths, circuits_per_depth, seed=5)
        extra_attrs = self._extra_paired_attrs()
        for depth_idx, circuits_at_depth in enumerate(design.circuit_lists):
            n = len(circuits_at_depth)
            self.assertEqual(len(design.euler_angles[depth_idx]), n)
            self.assertEqual(len(design.seq_index[depth_idx]), n)
            self.assertEqual(len(design.prep_index[depth_idx]), n)
            for attr in extra_attrs:
                self.assertEqual(len(getattr(design, attr)[depth_idx]), n)
            for circuit_idx, c in enumerate(circuits_at_depth):
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                self.assertTrue(np.array_equal(angles, euler_angles_from_circuit(c)))
                prep_idx = design.prep_index[depth_idx][circuit_idx]
                names = [layer.name for layer in c]
                self.assertEqual(names[0], 'rho%d' % prep_idx)
                self.assertEqual(names[-1], 'Mdefault')

    def test_ideal_composition(self):
        spinj = SpinJ(self.j)
        depths = [1, 2, 5]
        design = self.design_cls(self.j, depths, circuits_per_depth=4, seed=7)
        for depth_idx, depth in enumerate(design.depths):
            # only need to check one prep's circuit per sequence -- the angle array is
            # shared across all `dim` preps generated from the same sequence.
            for seq in range(design.circuits_per_depth):
                circuit_idx = seq * design.dim  # prep_index == 0 circuit for this sequence
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                self.assertEqual(angles.shape, (depth + 1, 3))
                net = _net_unitary(spinj, angles)
                target = self._target_unitary(spinj, angles)
                self.assertLess(distance_mod_phase(net, target), 1e-10)

    def _target_unitary(self, spinj, angles):
        raise NotImplementedError

    def test_truncate_to_lists(self):
        depths = [1, 2, 3]
        design = self.design_cls(self.j, depths, circuits_per_depth=2, seed=3)
        truncated = design.truncate_to_lists([0, 2])
        self.assertEqual(truncated.depths, [1, 3])
        self.assertEqual(len(truncated.circuit_lists), 2)
        self.assertEqual(truncated.circuit_lists[0], design.circuit_lists[0])
        self.assertEqual(truncated.circuit_lists[1], design.circuit_lists[2])
        self.assertEqual(truncated.euler_angles[0], design.euler_angles[0])
        self.assertEqual(truncated.euler_angles[1], design.euler_angles[2])
        self.assertEqual(truncated.seq_index[1], design.seq_index[2])
        self.assertEqual(truncated.prep_index[1], design.prep_index[2])
        for attr in self._extra_paired_attrs():
            self.assertEqual(getattr(truncated, attr)[0], getattr(design, attr)[0])
            self.assertEqual(getattr(truncated, attr)[1], getattr(design, attr)[2])

    def test_serialization_round_trip(self):
        depths = [1, 2]
        design = self.design_cls(self.j, depths, circuits_per_depth=2, seed=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            design.write(tmpdir)
            reloaded = self.design_cls.from_dir(tmpdir)
        self.assertEqual(design.circuit_lists, reloaded.circuit_lists)
        self.assertEqual(design.euler_angles, reloaded.euler_angles)
        self.assertEqual(design.seq_index, reloaded.seq_index)
        self.assertEqual(design.prep_index, reloaded.prep_index)
        self.assertEqual(design.j, reloaded.j)
        self.assertEqual(design.dim, reloaded.dim)
        self._check_extra_serialized_attrs(design, reloaded)

    def _check_extra_serialized_attrs(self, design, reloaded):
        pass


class TestSU2RBDesignHalfInt(_SU2RBDesignChecks, BaseCase):
    j = 0.5
    design_cls = SU2RBDesign

    def _target_unitary(self, spinj, angles):
        return np.eye(spinj.dim, dtype=complex)


class TestSU2RBDesignThreeHalves(_SU2RBDesignChecks, BaseCase):
    j = 1.5
    design_cls = SU2RBDesign

    def _target_unitary(self, spinj, angles):
        return np.eye(spinj.dim, dtype=complex)


class TestSU2CharacterRBDesignHalfInt(_SU2RBDesignChecks, BaseCase):
    j = 0.5
    design_cls = SU2CharacterRBDesign

    def _target_unitary(self, spinj, angles):
        return spinj.unitaries_from_angles(angles[0, 0], angles[0, 1], angles[0, 2])[0]

    def _extra_paired_attrs(self):
        return ["characters", "charcores"]

    def _check_extra_serialized_attrs(self, design, reloaded):
        self.assertEqual(design.characters, reloaded.characters)
        self.assertEqual(design.charcores, reloaded.charcores)


class TestSU2CharacterRBDesignThreeHalves(_SU2RBDesignChecks, BaseCase):
    j = 1.5
    design_cls = SU2CharacterRBDesign

    def _target_unitary(self, spinj, angles):
        return spinj.unitaries_from_angles(angles[0, 0], angles[0, 1], angles[0, 2])[0]

    def _extra_paired_attrs(self):
        return ["characters", "charcores"]

    def _check_extra_serialized_attrs(self, design, reloaded):
        self.assertEqual(design.characters, reloaded.characters)
        self.assertEqual(design.charcores, reloaded.charcores)


class TestSU2CharacterRBDesignExtras(BaseCase):
    """Character-design-specific checks that don't fit the shared mixin."""

    def test_invert_from(self):
        self.assertEqual(SU2RBDesign.invert_from, 0)
        self.assertEqual(SU2CharacterRBDesign.invert_from, 1)

    def test_characters_and_charcores_match_direct_evaluation(self):
        from pygsti.tools.su2tools import characters_from_euler_angles, charactercores_from_euler_angles

        j = 1.5
        design = SU2CharacterRBDesign(j, depths=[1, 4], circuits_per_depth=3, seed=42)
        dim = design.dim
        irrep_labels = np.arange(dim)
        for depth_idx in range(len(design.depths)):
            for circuit_idx in range(len(design.circuit_lists[depth_idx])):
                angles = np.array(design.euler_angles[depth_idx][circuit_idx])
                first_gate_angles = angles[0, :]
                expected_chars = characters_from_euler_angles(irrep_labels, first_gate_angles)
                expected_cores = charactercores_from_euler_angles(irrep_labels, first_gate_angles)
                actual_chars = np.array(design.characters[depth_idx][circuit_idx])
                actual_cores = np.array(design.charcores[depth_idx][circuit_idx])
                self.assertTrue(np.allclose(actual_chars, expected_chars, atol=1e-12))
                self.assertTrue(np.allclose(actual_cores, expected_cores, atol=1e-12))

    def test_depth_zero_raises(self):
        with self.assertRaises(ValueError):
            SU2RBDesign(0.5, depths=[0, 1], circuits_per_depth=2, seed=0)
        with self.assertRaises(ValueError):
            SU2CharacterRBDesign(0.5, depths=[0, 1], circuits_per_depth=2, seed=0)


# ---------------------------------------------------------------------------
# SU2RBDataSimulator (Phase 4)
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
    `spinj.dim`, *not* auto-cast to a multi-qubit space -- see the Task 3.0 spike
    notes' `default_space_for_udim` pitfall) whose `Gu` factory reproduces
    `spinj.unitaries_from_angles`, and whose `rho{ell}`/`Mdefault` preps/POVM match
    the SU2RBDesign circuit convention. Used to cross-validate SU2RBDataSimulator's
    noiseless probabilities against pyGSTi's standard model.probabilities() path.
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
    model.povms[POVM_NAME] = UnconstrainedPOVM(effects, evotype='default')

    def fn(args):
        alpha, beta, gamma = args
        return spinj.unitaries_from_angles(alpha, beta, gamma)[0]

    model.factories[GATE_NAME] = UnitaryOpFactory(fn, state_space, superop_basis=mx_basis, evotype='densitymx')
    return model


class TestSU2RBDataSimulatorNoiseless(BaseCase):

    def test_hand_composed_probabilities(self):
        j = 1.5
        spinj = SpinJ(j)
        design = SU2RBDesign(j, depths=[2, 4], circuits_per_depth=3, seed=1)
        sim = SU2RBDataSimulator(spinj)
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
        design = SU2RBDesign(j, depths=[1, 2], circuits_per_depth=2, seed=2)
        sim = SU2RBDataSimulator(j)
        data = sim.run(design)
        design_circuits = set(design.all_circuits_needing_data)
        dataset_circuits = set(data.dataset.keys())
        self.assertEqual(design_circuits, dataset_circuits)


class TestSU2RBDataSimulatorCrossValidation(BaseCase):
    """Phase 4 test (b): model.probabilities(circuit) agreement, pinning the circuit
    label convention to the simulator's semantics."""

    def test_matches_model_probabilities_j_half(self):
        self._check_j(0.5)

    def test_matches_model_probabilities_j_three_halves(self):
        self._check_j(1.5)

    def _check_j(self, j):
        spinj = SpinJ(j)
        model = _crossval_model(spinj)
        design = SU2RBDesign(j, depths=[1, 3], circuits_per_depth=2, seed=99)
        sim = SU2RBDataSimulator(spinj)
        data = sim.run(design)

        for circuits_at_depth in design.circuit_lists:
            for circuit in circuits_at_depth:
                model_probs = model.probabilities(circuit)
                sim_probs = _dataset_probs(data.dataset, circuit, spinj.dim)
                for ell in range(spinj.dim):
                    self.assertAlmostEqual(sim_probs[ell], model_probs[(str(ell),)], places=12)

    def test_matches_model_probabilities_character_design(self):
        j = 0.5
        spinj = SpinJ(j)
        model = _crossval_model(spinj)
        design = SU2CharacterRBDesign(j, depths=[1, 2], circuits_per_depth=2, seed=100)
        sim = SU2RBDataSimulator(spinj)
        data = sim.run(design)
        for circuits_at_depth in design.circuit_lists:
            for circuit in circuits_at_depth:
                model_probs = model.probabilities(circuit)
                sim_probs = _dataset_probs(data.dataset, circuit, spinj.dim)
                for ell in range(spinj.dim):
                    self.assertAlmostEqual(sim_probs[ell], model_probs[(str(ell),)], places=12)


class TestSU2RBDataSimulatorSpamSwap(BaseCase):

    def test_probabilities_from_compositions_matches_run(self):
        j = 1.5
        spinj = SpinJ(j)
        noise = jz_dephasing(spinj, 0.15, 1.0)
        sim = SU2RBDataSimulator(spinj, noise_channel=noise)
        design = SU2RBDesign(j, depths=[1, 3], circuits_per_depth=4, seed=3)
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

    def test_probabilities_from_compositions_character_design_unitary_noise(self):
        j = 0.5
        spinj = SpinJ(j)
        noise = jz_rotation(spinj, 0.3, 1.0)
        sim = SU2RBDataSimulator(spinj, noise_channel=noise)
        design = SU2CharacterRBDesign(j, depths=[2, 5], circuits_per_depth=3, seed=4)
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
        sim = SU2RBDataSimulator(spinj)
        design = SU2RBDesign(j, depths=[1, 2, 3], circuits_per_depth=2, seed=5)
        compositions = sim.compute_nonspam_compositions(design, depth_indices=[1])
        self.assertEqual(list(compositions.keys()), [1])
        self.assertEqual(compositions[1].shape, (2, spinj.dim ** 2, spinj.dim ** 2))

    def test_perturbed_spam_changes_probabilities(self):
        j = 0.5
        spinj = SpinJ(j)
        dim = spinj.dim
        sim = SU2RBDataSimulator(spinj)
        design = SU2RBDesign(j, depths=[3], circuits_per_depth=2, seed=6)
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
        sim = SU2RBDataSimulator(spinj)
        design = SU2RBDesign(j, depths=[1], circuits_per_depth=1, seed=7)
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


class TestSU2RBDataSimulatorShots(BaseCase):

    def test_shots_totals_and_reproducibility(self):
        j = 1.5
        spinj = SpinJ(j)
        design = SU2RBDesign(j, depths=[1, 2], circuits_per_depth=3, seed=7)

        sim_a = SU2RBDataSimulator(spinj, shots=1000, seed=42)
        sim_b = SU2RBDataSimulator(spinj, shots=1000, seed=42)
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
        # Noiseless SSRB circuits have deterministic (one-hot) outcome distributions,
        # which would make multinomial sampling insensitive to the seed; use a noisy
        # simulator so the per-shot outcome distribution is nontrivial.
        noise = jz_dephasing(spinj, 0.3)
        design = SU2RBDesign(j, depths=[3], circuits_per_depth=5, seed=8)
        sim_a = SU2RBDataSimulator(spinj, noise_channel=noise, shots=200, seed=1)
        sim_b = SU2RBDataSimulator(spinj, noise_channel=noise, shots=200, seed=2)
        data_a = sim_a.run(design)
        data_b = sim_b.run(design)
        any_different = any(
            dict(data_a.dataset[c].counts) != dict(data_b.dataset[c].counts)
            for c in design.circuit_lists[0]
        )
        self.assertTrue(any_different)


class TestSU2RBDataSimulatorCharacterShortcut(BaseCase):

    def test_shortcut_matches_long_path_noiseless(self):
        j = 1.5
        spinj = SpinJ(j)
        sim = SU2RBDataSimulator(spinj)
        design = SU2CharacterRBDesign(j, depths=[1, 4, 6], circuits_per_depth=3, seed=9)
        for depth_idx in range(len(design.depths)):
            for angles in sim._unique_sequences_at_depth(design, depth_idx):
                S_shortcut = sim._compose_shortcut(angles)
                S_full = sim._compose_full(angles, skip_first_noise=True)
                self.assertTrue(np.allclose(S_shortcut, S_full, atol=1e-10))

    def test_shortcut_used_only_when_noiseless_and_skipping(self):
        j = 0.5
        spinj = SpinJ(j)
        angles = np.array([[0.3, 0.5, 0.1], [1.1, 0.4, 2.0]])

        sim_noiseless = SU2RBDataSimulator(spinj)
        self.assertTrue(np.allclose(
            sim_noiseless._compose(angles, skip_first_noise=True),
            sim_noiseless._compose_shortcut(angles), atol=1e-12))

        noisy_sim = SU2RBDataSimulator(spinj, noise_channel=jz_dephasing(spinj, 0.2))
        # With noise present, the shortcut is not equivalent to the full path, so the
        # dispatcher must not take it.
        composed = noisy_sim._compose(angles, skip_first_noise=True)
        full = noisy_sim._compose_full(angles, skip_first_noise=True)
        self.assertTrue(np.allclose(composed, full, atol=1e-12))
        shortcut = noisy_sim._compose_shortcut(angles)
        self.assertFalse(np.allclose(composed, shortcut, atol=1e-6))


class TestSU2RBDataSimulatorNoisePlacement(BaseCase):
    """Pins the noise-insertion order and skip-first-noise semantics of
    `_compose_full` against an independently hand-built expectation (rather than
    comparing `_compose`-derived quantities to each other, as the noiseless-shortcut
    tests above do)."""

    def test_two_gate_noise_placement_and_skip_semantics(self):
        j = 1.5
        spinj = SpinJ(j)
        N = jz_dephasing(spinj, 0.25, power=1.0)
        sim = SU2RBDataSimulator(spinj, noise_channel=N)

        angles = np.array([[0.3, 0.5, 0.1], [1.2, 0.7, 2.1]])
        U0, U1 = spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
        S0 = unitary_to_std_process_mx(U0)
        S1 = unitary_to_std_process_mx(U1)

        # skip_first_noise=False: noise is inserted after *every* gate.
        expected_no_skip = N @ S1 @ N @ S0
        actual_no_skip = sim._compose_full(angles, skip_first_noise=False)
        self.assertTrue(np.allclose(actual_no_skip, expected_no_skip, atol=1e-12))

        # skip_first_noise=True: noise is skipped immediately after the first gate
        # only; it is still applied after the second (and every subsequent) gate.
        expected_skip = N @ S1 @ S0
        actual_skip = sim._compose_full(angles, skip_first_noise=True)
        self.assertTrue(np.allclose(actual_skip, expected_skip, atol=1e-12))

        # The two must differ (sanity check that this test isn't vacuous).
        self.assertFalse(np.allclose(expected_no_skip, expected_skip, atol=1e-6))


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


class TestSU2RBDataSimulatorConstruction(BaseCase):

    def test_accepts_spin_value_or_spinj(self):
        sim_from_j = SU2RBDataSimulator(1.5)
        sim_from_spinj = SU2RBDataSimulator(SpinJ(1.5))
        self.assertEqual(sim_from_j.dim, sim_from_spinj.dim)
        self.assertTrue(sim_from_j.is_noiseless)

    def test_bad_noise_channel_shape_raises(self):
        with self.assertRaises(ValueError):
            SU2RBDataSimulator(0.5, noise_channel=np.eye(3))

    def test_dim_mismatch_with_design_raises(self):
        sim = SU2RBDataSimulator(0.5)
        design = SU2RBDesign(1.5, depths=[1], circuits_per_depth=2, seed=10)
        with self.assertRaises(ValueError):
            sim.run(design)


# ---------------------------------------------------------------------------
# predicted_zero_noise_variance (Phase 5)
# ---------------------------------------------------------------------------

class TestPredictedZeroNoiseVariance(BaseCase):
    """
    Golden-table tests against the paper's Tables `variancewithk` and `variancewithj`
    (6-significant-figure values quoted in the plan / paper LaTeX source), plus the
    input-validation contract of `predicted_zero_noise_variance`.
    """

    def _assert_close_6sf(self, actual, golden):
        # The golden literals themselves are only quoted to 6 significant figures, so
        # allow a couple of ULPs of rounding slop on top of that (verified against a
        # from-scratch double-precision recomputation of both paper tables during this
        # phase's development; see the phase 5 completion notes).
        self.assertAlmostEqual(actual, golden, delta=abs(golden) * 2e-5 + 1e-9)

    def test_ssrb_is_always_zero(self):
        for j, k in [(0.5, 0), (0.5, 1), (3.5, 0), (3.5, 7)]:
            self.assertEqual(predicted_zero_noise_variance(j, k, 'ssrb'), 0.0)

    def test_variancewithk_table_j_seven_halves(self):
        # Table `variancewithk`: j = 7/2, k = 0..7. The optimal ell for the physical-
        # SPAM (chiRB/R1RB) columns is as given in the paper text just above the table.
        j = 3.5
        ells_for_k = {0: 3.5, 1: 3.5, 2: 3.5, 3: 1.5, 4: 2.5, 5: 2.5, 6: 1.5, 7: 0.5}
        golden = {
            # k: (chiRB, R1RB, SSchiRB, SSR1RB)
            0: (7, 7, 0.0, 0.0),
            1: (28.6816, 7.52245, 1.07619, 0.269048),
            2: (91.8386, 12.5807, 3.23842, 0.540816),
            3: (308.139, 42.3744, 6.15572, 0.773292),
            4: (268.103, 21.0241, 10.4498, 1.02387),
            5: (514.734, 32.779, 15.668, 1.28994),
            6: (404.56, 23.2173, 23.0531, 1.62223),
            7: (381.656, 21.6442, 34.0697, 2.11888),
        }
        for k, (chi, r1, sschi, ssr1) in golden.items():
            ell = ells_for_k[k]
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'chirb', ell=ell), chi)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'r1rb', ell=ell), r1)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'sschirb'), sschi)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'ssr1rb'), ssr1)

    def test_variancewithj_table(self):
        # Table `variancewithj`: k = 2j fixed, j = 0..7/2. Optimal ell is 0 for integer
        # j and 1/2 for half-integer j (paper text just above the table).
        golden = {
            # j: (chiRB, R1RB, SSchiRB, SSR1RB)
            0.0: (0, 0, 0, 0),
            0.5: (23, 5, 4, 1),
            1.0: (25.25, 4.89286, 8.66667, 1.40476),
            1.5: (91.1811, 9.9465, 13.408, 1.63867),
            2.0: (95.25, 11.163, 18.4047, 1.80578),
            2.5: (209.672, 15.5894, 23.5132, 1.9322),
            3.0: (215.636, 18.0822, 28.7441, 2.03407),
            3.5: (381.656, 21.6442, 34.0697, 2.11888),
        }
        for j, (chi, r1, sschi, ssr1) in golden.items():
            k = round(2 * j)
            ell = 0.0 if (round(2 * j) % 2 == 0) else 0.5
            if k == 0:
                self.assertEqual(predicted_zero_noise_variance(j, k, 'sschirb'), 0.0)
                self.assertEqual(predicted_zero_noise_variance(j, k, 'ssr1rb'), 0.0)
                self.assertEqual(chi, 0)
                self.assertEqual(r1, 0)
                continue
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'chirb', ell=ell), chi)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'r1rb', ell=ell), r1)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'sschirb'), sschi)
            self._assert_close_6sf(predicted_zero_noise_variance(j, k, 'ssr1rb'), ssr1)

    def test_bad_variant_raises(self):
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'not-a-variant')

    def test_bad_k_raises(self):
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 4, 'sschirb')  # 2j == 3 < 4
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, -1, 'sschirb')

    def test_ell_required_for_physical_spam_variants(self):
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'chirb')  # missing ell
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'r1rb')  # missing ell

    def test_ell_disallowed_for_synthetic_spam_and_ssrb_variants(self):
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'ssrb', ell=1.5)
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'sschirb', ell=1.5)
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1.5, 1, 'ssr1rb', ell=1.5)

    def test_invalid_ell_raises(self):
        # ell=2 is not one of SpinJ(1).spins == [1, 0, -1]; wignersymbols.clebsch_gordan
        # would otherwise silently treat this as an ordinary |m|<=j selection-rule
        # violation (returning 0.0), which would then divide-by-zero downstream --
        # this should instead be reported as a caller error.
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1, 1, 'chirb', ell=2)
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1, 1, 'r1rb', ell=-2)
        # Also not a valid half-integer at all.
        with self.assertRaises(ValueError):
            predicted_zero_noise_variance(1, 1, 'chirb', ell=0.3)

    def test_genuinely_vanishing_mean_returns_inf(self):
        # j=1, k=1, ell=0 is a *valid* Jz eigenstate, but M[1, 0] = sqrt(1/3) *
        # <1 0; 1 0 | 1 0> is exactly 0 (a Clebsch-Gordan selection rule, not an
        # out-of-range ell), so Eq. normalizedvariance's mean-normalized variance
        # (which divides by M[k,ell]**4) is genuinely infinite rather than an
        # arithmetic-error artifact.
        self.assertEqual(predicted_zero_noise_variance(1, 1, 'chirb', ell=0), float('inf'))
        self.assertEqual(predicted_zero_noise_variance(1, 1, 'r1rb', ell=0), float('inf'))


# ---------------------------------------------------------------------------
# SyntheticSPAMRB / SyntheticSPAMRBResults (Phase 5)
# ---------------------------------------------------------------------------

class TestSyntheticSPAMRBNoiseless(BaseCase):

    def test_noiseless_decays_and_rates(self):
        # Zero noise: P is exactly the identity for every circuit/prep (the net ideal
        # composition of an SSRB circuit is the identity), so X_k = diag(M M^T)[k] = 1
        # exactly (M is orthogonal) for every sequence/depth -- the fitted decays should
        # be (numerically) exactly 1, and the recovered rates exactly e_0 (F's row/
        # column 0 is all ones, so solve(F, ones) = e_0).
        j = 1.5
        spinj = SpinJ(j)
        sim = SU2RBDataSimulator(spinj)
        design = SU2RBDesign(j, depths=[1, 2, 3, 5, 8], circuits_per_depth=5, seed=3)
        data = sim.run(design)
        results = SyntheticSPAMRB().run(data)

        self.assertTrue(np.allclose(results.decays, 1.0, atol=1e-6))
        expected_p = np.zeros(spinj.dim)
        expected_p[0] = 1.0
        self.assertTrue(np.allclose(results.rates, expected_p, atol=1e-6))

    def test_rates_dataframe_and_variance_diagnostic(self):
        j = 0.5
        spinj = SpinJ(j)
        sim = SU2RBDataSimulator(spinj)
        # >=3 depths: with exactly 2, the 2-parameter (a, f) fit always has zero
        # residual degrees of freedom, which scipy.optimize.curve_fit treats as an
        # indeterminate-covariance case (a warning promoted to a fit failure by this
        # module -- see test_failed_irrep_fit_warns) regardless of data quality.
        design = SU2RBDesign(j, depths=[1, 2, 3], circuits_per_depth=3, seed=1)
        data = sim.run(design)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            results = SyntheticSPAMRB().run(data)

        df = results.rates_dataframe()
        self.assertEqual(len(df), spinj.dim)
        self.assertEqual(list(df['irrep']), list(range(spinj.dim)))
        for col in ('decay_f', 'decay_f_stderr', 'rate_p', 'rate_p_stderr'):
            self.assertIn(col, df.columns)

        diag = results.variance_diagnostic()
        self.assertEqual(set(diag.keys()), set(range(spinj.dim)))
        for k, (predicted, empirical) in diag.items():
            # SyntheticSPAMRB._variance_variant == 'ssrb', whose predicted variance is
            # always 0.
            self.assertEqual(predicted, 0.0)
            self.assertGreaterEqual(empirical, 0.0)

    def test_isinstance_results(self):
        j = 0.5
        # >=3 depths: with exactly 2, the 2-parameter (a, f) fit always has zero
        # residual degrees of freedom, which is treated as a fit failure (see
        # test_rates_dataframe_and_variance_diagnostic's comment / test_failed_irrep_fit_warns).
        design = SU2RBDesign(j, depths=[1, 2, 3], circuits_per_depth=2, seed=0)
        data = SU2RBDataSimulator(j).run(design)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            results = SyntheticSPAMRB().run(data)
        self.assertIsInstance(results, SyntheticSPAMRBResults)

    def test_failed_irrep_fit_warns(self):
        # A single-depth design under-determines the 2-parameter (a, f) fit for every
        # irrep (scipy.optimize.curve_fit cannot estimate a covariance from one data
        # point), which should surface as a UserWarning naming the failed irreps --
        # not a silent all-nan `rates`/`rates_covariance` (F mixes every irrep
        # together, so one failed fit poisons all of them).
        j = 0.5
        design = SU2RBDesign(j, depths=[1], circuits_per_depth=2, seed=0)
        data = SU2RBDataSimulator(j).run(design)
        with self.assertWarns(UserWarning) as cm:
            results = SyntheticSPAMRB().run(data)
        self.assertIn('irrep', str(cm.warning))
        self.assertTrue(np.all(np.isnan(results.rates)))


class TestSyntheticSPAMRBAnalyticEndToEnd(BaseCase):
    """
    Phase 5's analytic end-to-end test: simulate SSRB with a known gate-independent
    noise channel and exact (shots=None) probabilities, and check the fitted per-irrep
    decays against the closed-form irrep fidelities `f_k = Tr(P_k . Lambda) / (2k+1)`
    (`Lambda` the noise channel's standard-basis superoperator, `P_k` from
    `SpinJ.irrep_stdmx_projectors`) -- the standard RB "twirl" formula.

    A weak noise strength (`gamma = 2e-4`) is used deliberately: the per-circuit
    zero-noise variance (paper Eq. `SSvariance`; `SyntheticSPAMRB._variance_variant ==
    'ssrb'` is exactly 0 at zero noise) grows continuously from 0 as gamma turns on, so
    at weak noise the finite-circuit (N=100) sampling scatter around the analytic decay
    curve is itself O(gamma) and comfortably below a 1e-6 absolute tolerance; at the
    gamma ~ 0.05 scale used elsewhere in this test module for cross-validation, the
    same N=100/depths-to-30 budget only resolves the decays to ~1e-4 (this was checked
    empirically during development -- see the phase 5 completion notes -- and is
    consistent with the paper's point that synthetic RB's variance is dominated by
    circuit-sampling scatter rather than by measurement statistics).
    """

    def _run_and_check(self, j, gamma=2e-4, n_circuits=100, max_depth=30, seed=42, atol=1e-6):
        spinj = SpinJ(j)
        dim = spinj.dim
        noise = jz_dephasing(spinj, gamma, power=1.0)
        sim = SU2RBDataSimulator(spinj, noise_channel=noise, shots=None, seed=1)
        depths = list(range(1, max_depth + 1))
        design = SU2RBDesign(j, depths, circuits_per_depth=n_circuits, seed=seed)
        data = sim.run(design)
        results = SyntheticSPAMRB().run(data)

        analytic_f = np.array([
            np.trace(spinj.irrep_stdmx_projectors[k] @ sim._noise_superop).real / (2 * k + 1)
            for k in range(dim)
        ])
        self.assertTrue(np.allclose(results.decays, analytic_f, atol=atol),
                         msg="decays=%s analytic=%s" % (results.decays, analytic_f))

        # k=0 (the trivial irrep) fits to (numerically) exactly 1 regardless of noise:
        # trace preservation makes its series exactly 1 for every single circuit.
        self.assertAlmostEqual(results.decays[0], 1.0, delta=1e-8)

        # Recovered rates satisfy the f = F @ p round trip.
        F = spinj.decay_recoupling_matrix
        self.assertTrue(np.allclose(F @ results.rates, results.decays, atol=1e-10))

        # Covariance propagation: right shape, symmetric, positive semidefinite.
        cov = results.rates_covariance
        self.assertEqual(cov.shape, (dim, dim))
        self.assertTrue(np.allclose(cov, cov.T))
        eigvals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(eigvals >= -1e-12), msg="eigvals=%s" % eigvals)

        return results

    def test_j_three_halves(self):
        self._run_and_check(1.5)

    def test_j_seven_halves(self):
        self._run_and_check(3.5)
