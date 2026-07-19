"""
Tests for pygsti.protocols.su2rb (Phase 3: SU(2) synthetic SPAM RB circuits and
experiment designs).
"""
import tempfile

import numpy as np

from pygsti.circuits.circuit import Circuit
from pygsti.protocols.su2rb import (
    SU2RBDesign,
    SU2CharacterRBDesign,
    circuit_from_euler_angles,
    euler_angles_from_circuit,
    GATE_NAME,
)
from pygsti.tools.su2tools import SpinJ, distance_mod_phase

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
