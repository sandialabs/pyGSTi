import copy
import pickle
import unittest

from pygsti.circuits import circuit
from pygsti.baseobjs import Label, CircuitLabel
from pygsti.processors import ProcessorPack
from pygsti.tools import symplectic
from ..util import BaseCase


class CircuitTester(BaseCase):
    def test_construct_from_empty(self):
        # Test initializing a circuit from an empty circuit.
        c = circuit.Circuit(num_lines=5)
        self.assertEqual(c.depth, 0)
        self.assertEqual(c.size, 0)
        self.assertEqual(c.num_lines, 5)
        self.assertEqual(c.line_labels, tuple(range(5)))

        c = circuit.Circuit(layer_labels=[], num_lines=5)
        self.assertEqual(c.depth, 0)
        self.assertEqual(c.size, 0)
        self.assertEqual(c.num_lines, 5)
        self.assertEqual(c.line_labels, tuple(range(5)))

    def test_construct_from_label(self):
        # Test initializing a circuit from a non-empty circuit that is a list
        # containing Label objects. Also test that it can have non-integer line_labels
        # and a different identity identifier.
        labels = [Label('Gi', 'Q0'), Label('Gp', 'Q8')]
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1', 'Q8', 'Q12'])
        # Not parallelized by default, so will be depth 2.
        self.assertEqual(c.depth, 2)
        self.assertEqual(c.size, 2)
        self.assertEqual(c.num_lines, 4)
        self.assertEqual(c.line_labels, ('Q0', 'Q1', 'Q8', 'Q12'))

    def test_construct_from_label_parallelized(self):
        # Do again with parallelization
        labels = [Label('Gi', 'Q0'), Label('Gp', 'Q8')]
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1', 'Q8'])
        c = c.parallelize()
        self.assertEqual(c.depth, 1)
        self.assertEqual(c.size, 2)

    def test_construction_label_conversion(self):
        # XXX what is tested here that is not covered by other tests?  EGN: this is more of a use case for when this input is a *nested* tuple.
        # Check that parallel operation labels get converted to circuits properly
        opstr = circuit.Circuit(((('Gx', 0), ('Gy', 1)), ('Gcnot', 0, 1)))
        c = circuit.Circuit(layer_labels=opstr, num_lines=2)
        self.assertEqual(c._labels, (Label((('Gx', 0), ('Gy', 1))), Label('Gcnot', (0, 1))))

    def test_construct_from_oplabels(self):
        # Now repeat the read-in with no parallelize, but a list of lists of oplabels
        labels = [[Label('Gi', 'Q0'), Label('Gp', 'Q8')], [Label('Gh', 'Q1'), Label('Gp', 'Q12')]]
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1', 'Q8', 'Q12'])
        self.assertLess(0, c.depth)

    def test_construct_from_label_without_qubit_labels(self):
        # Check we can read-in a circuit that has no qubit labels: enforces them to be on
        # all of the lines.
        labels = circuit.Circuit(None, stringrep="Gx^2GyGxGi")
        c = circuit.Circuit(layer_labels=labels, num_lines=1)
        self.assertEqual(c.depth, 5)

    def test_construct_from_label_editable(self):
        # Check that we can create a circuit from a string and that we end up with
        # the correctly structured circuit.
        labels = circuit.Circuit(None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'], editable=True)
        c2 = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'], editable=False)
        self.assertEqual(c.tup, c2.tup)
        self.assertEqual(c.depth, 5)
        self.assertEqual(c.size, 8)

    def test_construct_from_circuit(self):
        # Check we can init from another circuit
        labels = circuit.Circuit(None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'], editable=True)
        cnew = circuit.Circuit(c)
        self.assertEqual(cnew, c)

    def test_create_circuitlabel(self):
        # test automatic creation of "power" subcircuits when needed
        Gi = circuit.Circuit(None, stringrep='Gi(Gx)^256000', editable=True, expand_subcircuits=False)
        self.assertTrue(isinstance(Gi.tup[1], CircuitLabel))

        cl = Gi.tup[1]
        self.assertEqual(str(cl), "(Gx)^256000")
        self.assertEqual(cl.components, ('Gx',))
        self.assertEqual(cl.reps, 256000)
        self.assertEqual(Gi.tup, ('Gi', CircuitLabel(name='', tup_of_layers=('Gx',),
                                                     state_space_labels=None, reps=256000)))

    def test_expand_and_factorize_circuitlabel(self):
        c = circuit.Circuit(None, stringrep='Gi(Gx:1)^2', num_lines=3, editable=True, expand_subcircuits=False)
        c[1, 0] = "Gx"
        self.assertEqual(c.tup, ('Gi', (CircuitLabel('', [('Gx', 1)], (1,), 2), ('Gx', 0))) + ('@', 0, 1, 2))

        c_expanded = c.expand_subcircuits()
        c.expand_subcircuits_inplace()
        self.assertEqual(c, c_expanded)
        self.assertEqual(c.tup, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gx', 1)) + ('@', 0, 1, 2))
        self.assertEqual(c, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gx', 1)))   # `c` compares vs. labels when RHS is not a Circuit

        c2 = circuit.Circuit(None, stringrep='GiGxGxGxGxGy', editable=True)
        self.assertEqual(c2, ('Gi', 'Gx', 'Gx', 'Gx', 'Gx', 'Gy'))

        c2.factorize_repetitions_inplace()
        self.assertEqual(c2, ('Gi', CircuitLabel('', ['Gx'], None, 4), 'Gy'))

    def test_circuitlabel_inclusion(self):
        c = circuit.Circuit(None, stringrep="GxGx(GyGiGi)^2", expand_subcircuits=False)
        self.assertTrue('Gi' in c)
        self.assertEqual(['Gi' in layer for layer in c], [False, False, True])

        c = circuit.Circuit(None, stringrep="Gx:0[Gx:0(Gy:1GiGi)^2]", num_lines=2, expand_subcircuits=False)
        self.assertTrue('Gi' in c)
        self.assertEqual(['Gi' in layer for layer in c], [False, True])

    def test_circuit_str_is_updated(self):
        #Test that .str is updated
        c = circuit.Circuit(None, stringrep="GxGx(GyGiGi)^2", editable=True)
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "GxGxGyGiGiGyGiGi")

        c.delete_layers(slice(1, 4))
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "GxGiGyGiGi")
        c.done_editing()
        self.assertEqual(c.str, "GxGiGyGiGi")

        c = circuit.Circuit('Gi')
        c = c.copy(editable=True)
        self.assertEqual(c.str, "Gi")
        c.replace_gatename_inplace('Gi', 'Gx')
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "Gx")

    def test_circuit_exponentiation(self):
        circuit.Circuit.default_expand_subcircuits = False
        try:
            Gi = circuit.Circuit("Gi")
            Gy = circuit.Circuit("Gy")

            c = Gi + Gy**2048
            # more a label test? - but tests circuit exponentiation
            self.assertEqual(c[1].to_native(), ('', None, 2048, 'Gy'))
        finally:
            circuit.Circuit.default_expand_subcircuits = True

    def test_circuit_as_label(self):
        #test Circuit -> CircuitLabel conversion w/exponentiation
        c1 = circuit.Circuit(None, stringrep='Gi[][]', num_lines=4, editable=True)
        c2 = circuit.Circuit(None, stringrep='[Gx:0Gx:1][Gy:1]', num_lines=2)

        #Insert the 2Q circuit c2 into the 4Q circuit c as an exponentiated block (so c2 is exponentiated as well)
        c = c1.copy()
        c[1, 0:2] = c2.to_label(nreps=2)

        tup = ('Gi', ('', (0, 1), 2, (('Gx', 0), ('Gx', 1)), ('Gy', 1)), ()) + ('@', 0, 1, 2, 3)
        self.assertEqual(c.tup, tup)  # can't compare with `c` b/c comparison with non-Circuits compares against labels only
        self.assertEqual(c.num_layers, 3)
        self.assertEqual(c.depth, 6)
        # Qubit 0 ---|Gi|-||([Gx:0Gx:1]Gy:1)^2||-| |---
        # Qubit 1 ---|Gi|-||([Gx:0Gx:1]Gy:1)^2||-| |---
        # Qubit 2 ---|Gi|-|                    |-| |---
        # Qubit 3 ---|Gi|-|                    |-| |---

        c = c1.copy()
        c[1, 0:2] = c2  # special behavior: c2 is converted to a label to cram it into a single layer
        self.assertEqual(c.tup, ('Gi', ('', (0, 1), 1, (('Gx', 0), ('Gx', 1)), ('Gy', 1)), ()) + ('@', 0, 1, 2, 3))
        self.assertEqual(c, ('Gi', ('', (0, 1), 1, (('Gx', 0), ('Gx', 1)), ('Gy', 1)), ()))
        self.assertEqual(c.num_layers, 3)
        self.assertEqual(c.depth, 4)

        # Qubit 0 ---|Gi|-||([Gx:0Gx:1]Gy:1)||-| |---
        # Qubit 1 ---|Gi|-||([Gx:0Gx:1]Gy:1)||-| |---
        # Qubit 2 ---|Gi|-|                  |-| |---
        # Qubit 3 ---|Gi|-|                  |-| |---

        c = c1.copy()
        c[(1, 2), 0:2] = c2  # writes into described block
        self.assertEqual(c.tup, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy', 1)) + ('@', 0, 1, 2, 3))
        self.assertEqual(c, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy', 1)))
        self.assertEqual(c.num_layers, 3)
        self.assertEqual(c.depth, 3)
        # Qubit 0 ---|Gi|-|Gx|-|  |---
        # Qubit 1 ---|Gi|-|Gx|-|Gy|---
        # Qubit 2 ---|Gi|-|  |-|  |---
        # Qubit 3 ---|Gi|-|  |-|  |---

        c = c1.copy()
        c[(1, 2), 0:2] = c2.to_label().components  # same as above, but more roundabout
        self.assertEqual(c.tup, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy', 1)) + ('@', 0, 1, 2, 3))
        self.assertEqual(c, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy', 1)))
        self.assertEqual(c.num_layers, 3)
        self.assertEqual(c.depth, 3)

    def test_empty_tuple_makes_idle_layer(self):
        c = circuit.Circuit(['Gi', Label(())])
        self.assertEqual(len(c), 2)

    def test_replace_with_idling_line(self):
        c = circuit.Circuit([('Gcnot', 0, 1)], editable=True)
        c.replace_with_idling_line_inplace(0)
        self.assertEqual(c.tup, ((),) + ('@', 0, 1))
        self.assertEqual(c, ((),))

    def test_to_pythonstr(self):
        mdl = circuit.Circuit(None, stringrep="Gx^3Gy^2GxGz")

        op_labels = (Label('Gx'), Label('Gy'), Label('Gz'))
        pystr = mdl.to_pythonstr(op_labels)
        self.assertEqual(pystr, "AAABBAC")

        gs2_tup = circuit.Circuit.from_pythonstr(pystr, op_labels)
        self.assertEqual(gs2_tup, tuple(mdl))

    def test_raise_on_bad_construction(self):
        with self.assertRaises(ValueError):
            circuit.Circuit(('Gx', 'Gx'), stringrep="GxGy", check=True)  # mismatch
        with self.assertRaises(ValueError):
            circuit.Circuit(None)
        with self.assertRaises(ValueError):
            circuit.Circuit(('foobar',), stringrep="foobar", check=True)  # lexer illegal character


class CircuitMethodTester(BaseCase):
    def setUp(self):
        self.labels = circuit.Circuit(None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1")
        self.c = circuit.Circuit(layer_labels=self.labels, line_labels=['Q0', 'Q1'], editable=True)

    def test_insert_labels_into_layers_with_nonidle_qubits(self):
        # Test inserting a gate when the relevant qubits aren't
        # idling at that layer
        self.c.insert_labels_into_layers_inplace([Label('Gx', 'Q0')], 2)
        self.assertEqual(self.c.size, 9)
        self.assertEqual(self.c.depth, 6)
        self.assertEqual(self.c[2, 'Q0'], Label('Gx', 'Q0'))

    def test_insert_labels_into_layers_with_idle_qubits(self):
        # Test inserting a gate when the relevant qubits are
        # idling at that layer -- depth shouldn't increase
        self.c[2, 'Q1'] = Label('Gx', 'Q1')
        self.assertEqual(self.c.size, 8)
        self.assertEqual(self.c.depth, 5)
        self.assertEqual(self.c[2, 'Q1'], Label('Gx', 'Q1'))

    def test_insert_layer(self):
        # Test layer insertion
        layer = [Label('Gx', 'Q1'), ]
        self.c.insert_layer_inplace(layer, 1)
        self.assertEqual(self.c.size, 9)
        self.assertEqual(self.c.depth, 6)
        self.assertEqual(self.c[1], Label('Gx', 'Q1'))
        self.c.insert_layer_inplace([], 1)
        self.assertTrue(len(self.c[1, ('Q0', 'Q1')].components) == 0)
        self.assertTrue(len(self.c[1, 'Q0'].components) == 0)
        self.assertFalse(len(self.c[2, 'Q1'].components) == 0)
        self.assertFalse(self.c._is_line_idling('Q1'))
        self.c._append_idling_lines(['Q3'])
        self.assertFalse(self.c._is_line_idling('Q0'))
        self.assertFalse(self.c._is_line_idling('Q1'))
        self.assertTrue(self.c._is_line_idling('Q3'))

    def test_replace_layer_with_layer(self):
        # Test replacing a layer with a layer.
        newlayer = [Label('Gx', 'Q0')]
        self.c[1] = newlayer
        self.assertEqual(self.c.depth, 5)

    def test_replace_layer_with_circuit(self):
        # Test replacing a layer with a circuit
        self.c.replace_layer_with_circuit_inplace(self.c.copy(), 1)
        self.assertEqual(self.c.depth, 2 * 5 - 1)

    def test_delete_layers(self):
        # Test layer deletion
        layer = [Label('Gx', 'Q1'), ]
        c_copy = self.c.copy()
        c_copy.insert_layer_inplace(layer, 1)
        c_copy.delete_layers([1])
        self.assertEqual(self.c, c_copy)

    def test_insert_circuit_label_collision(self):
        # Test inserting a circuit when they are over the same labels.
        c_copy = self.c.copy()
        self.c.insert_circuit_inplace(c_copy, 2)
        self.assertTrue(Label('Gx', 'Q0') in self.c[2].components)

    def test_insert_circuit_with_qubit_superset(self):
        # Test insert a circuit that is over *more* qubits but which has the additional
        # lines idling.
        c2 = circuit.Circuit(layer_labels=self.labels, line_labels=['Q0', 'Q1', 'Q2', 'Q3'])
        self.c.insert_circuit_inplace(c2, 0)
        self.assertEqual(self.c.line_labels, ('Q0', 'Q1'))
        self.assertEqual(self.c.num_lines, 2)

    def test_insert_circuit_with_qubit_subset(self):
        # Test inserting a circuit that is on *less* qubits.
        c2 = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0', ])
        self.c.insert_circuit_inplace(c2, 1)
        self.assertEqual(self.c.line_labels, ('Q0', 'Q1'))
        self.assertEqual(self.c.num_lines, 2)

    def test_append_circuit(self):
        # Test appending
        c2 = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0', ], editable=True)
        self.c.append_circuit_inplace(c2)
        self.assertEqual(self.c.depth, 6)
        self.assertEqual(self.c[5, 'Q0'], Label('Gx', 'Q0'))

    def test_prefix_circuit(self):
        c2 = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0', ], editable=True)
        self.c.prefix_circuit_inplace(c2)
        self.assertEqual(self.c.depth, 6)
        self.assertEqual(self.c[0, 'Q0'], Label('Gx', 'Q0'))

    def test_tensor_circuit(self):
        # Test tensoring circuits of same length
        gatestring2 = circuit.Circuit(None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3")
        c2 = circuit.Circuit(layer_labels=gatestring2, line_labels=['Q2', 'Q3'])
        self.c.tensor_circuit_inplace(c2)
        self.assertEqual(self.c.depth, max(self.c.depth, c2.depth))
        self.assertEqual(self.c[:, 'Q2'], c2[:, 'Q2'])

    def test_tensor_circuit_with_shorter(self):
        # Test tensoring circuits where the inserted circuit is shorter
        gatestring2 = circuit.Circuit(None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3")
        c2 = circuit.Circuit(layer_labels=gatestring2, line_labels=['Q2', 'Q3'])
        self.c.tensor_circuit_inplace(c2, line_order=['Q1', 'Q3', 'Q0', 'Q2'])
        self.assertEqual(self.c.depth, max(self.c.depth, c2.depth))

    def test_tensor_circuit_with_longer(self):
        # Test tensoring circuits where the inserted circuit is shorter
        gatestring2 = circuit.Circuit(None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3Gy:Q2")
        c2 = circuit.Circuit(layer_labels=gatestring2, line_labels=['Q2', 'Q3'])
        self.c.tensor_circuit_inplace(c2)
        self.assertEqual(self.c.depth, max(self.c.depth, c2.depth))

    def test_replace_gatename_inplace(self):
        # Test changing a gate name
        self.c.replace_gatename_inplace('Gx', 'Gz')
        labels = circuit.Circuit(None, stringrep="[Gz:Q0Gy:Q1]^2[Gy:Q0Gz:Q1]Gi:Q0Gi:Q1")
        c2 = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'])
        self.assertEqual(self.c, c2)

    def test_change_gate_library(self):
        # Change gate library using an ordinary dict with every gate as a key. (we test
        # changing gate library using a CompilationLibrary elsewhere in the tests).
        labels = circuit.Circuit(None, stringrep="[Gz:Q0Gy:Q1][Gy:Q0Gz:Q1]Gz:Q0Gi:Q1")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'], editable=True)
        comp = {}
        comp[Label('Gz', 'Q0')] = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0'])
        comp[Label('Gy', 'Q0')] = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0'])
        comp[Label('Gz', 'Q1')] = circuit.Circuit(layer_labels=[Label('Gx', 'Q1')], line_labels=['Q1'])
        comp[Label('Gy', 'Q1')] = circuit.Circuit(layer_labels=[Label('Gx', 'Q1')], line_labels=['Q1'])
        comp[Label('Gi', 'Q1')] = circuit.Circuit(layer_labels=[Label('Gi', 'Q1')], line_labels=['Q1'])

        c.change_gate_library(comp)
        self.assertTrue(Label('Gx', 'Q0') in c[0].components)

    def test_change_gate_library_missing_gates(self):
        # Change gate library using a dict with some gates missing
        comp = {}
        comp[Label('Gz', 'Q0')] = circuit.Circuit(layer_labels=[Label('Gx', 'Q0')], line_labels=['Q0'])
        self.c.change_gate_library(comp, allow_unchanged_gates=True)
        self.assertTrue(Label('Gx', 'Q0') in self.c[0].components)  # c.layer(0)
        self.assertTrue(Label('Gy', 'Q1') in self.c[0].components)

    def test_map_state_space_labels_inplace(self):
        # Test we can change the labels of the lines.
        self.c.map_state_space_labels_inplace({'Q0': 0, 'Q1': 1})
        self.assertEqual(self.c.line_labels, (0, 1))
        self.assertEqual(self.c[0, 0].qubits[0], 0)

    def test_reorder_lines(self):
        # Check we can re-order wires
        self.c.map_state_space_labels_inplace({'Q0': 0, 'Q1': 1})
        self.c.reorder_lines_inplace([1, 0])
        self.assertEqual(self.c.line_labels, (1, 0))

    def test_append_and_delete_idling_lines(self):
        # Test deleting and inserting idling wires.
        self.c._append_idling_lines(['Q2'])
        self.assertEqual(self.c.line_labels, ('Q0', 'Q1', 'Q2'))
        self.assertEqual(self.c.num_lines, 3)
        self.c.delete_idling_lines_inplace()
        self.assertEqual(self.c.line_labels, ('Q0', 'Q1'))
        self.assertEqual(self.c.num_lines, 2)

    def test_circuit_reverse(self):
        # Test circuit reverse.
        op1 = self.c[0, 'Q0']
        self.c.reverse_inplace()
        op2 = self.c[-1, 'Q0']
        self.assertEqual(op1, op2)

    def test_twoQgate_count(self):
        self.assertEqual(self.c.two_q_gate_count(), 0)
        labels = circuit.Circuit(None, stringrep="[Gcnot:Q0:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'])
        self.assertEqual(c.two_q_gate_count(), 2)

    def test_multiQgate_count(self):
        self.assertEqual(self.c.num_multiq_gates, 0)
        labels = circuit.Circuit(None, stringrep="[Gccnot:Q0:Q1:Q2]^2[Gccnot:Q0:Q1]Gi:Q0Gi:Q1")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1', 'Q2'])
        self.assertEqual(c.num_multiq_gates, 3)

    def test_to_string(self):
        test_s = "Qubit Q0 ---|Gx|-|Gx|-|Gy|-|Gi|-|  |---\nQubit Q1 ---|Gy|-|Gy|-|Gx|-|  |-|Gi|---\n"
        s = str(self.c)
        self.assertEqual(test_s, s)

    def test_compress_depth(self):
        ls = [Label('H', 1), Label('P', 1), Label('P', 1), Label(()), Label('CNOT', (2, 3))]
        ls += [Label('HP', 1), Label('PH', 1), Label('CNOT', (1, 2))]
        ls += [Label(()), Label(()), Label('CNOT', (1, 2))]
        labels = circuit.Circuit(ls)
        c = circuit.Circuit(layer_labels=labels, num_lines=4, editable=True)
        c.compress_depth_inplace(verbosity=0)
        self.assertEqual(c.depth, 7)
        # Get a dictionary that relates H, P gates etc.
        oneQrelations = symplectic.one_q_clifford_symplectic_group_relations()
        c.compress_depth_inplace(one_q_gate_relations=oneQrelations)
        self.assertEqual(c.depth, 3)

    @unittest.skip("unused (remove?)")
    def test_predicted_error_probability(self):
        # Test the error-probability prediction method
        labels = circuit.Circuit(None, stringrep="[Gx:Q0][Gi:Q0Gi:Q1]")
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q0', 'Q1'])
        infidelity_dict = {}
        infidelity_dict[Label('Gi', 'Q0')] = 0.7
        infidelity_dict[Label('Gi', 'Q1')] = 0.9
        infidelity_dict[Label('Gx', 'Q0')] = 0.8
        infidelity_dict[Label('Gx', 'Q2')] = 0.9

        # TODO fix
        epsilon = c.predicted_error_probability(infidelity_dict)
        self.assertLess(abs(epsilon - (1 - (1 - 0.7) * (1 - 0.8) * (1 - 0.9)**2)), 10**-10)

    def test_convert_to_quil(self):
        # Quil string with setup, each layer, and block_between_layers=True (current default)
        quil_str = """DECLARE ro BIT[2]
RESET
PRAGMA INITIAL_REWIRING "NAIVE"
I 1
I 2
PRAGMA PRESERVE_BLOCK
PRAGMA END_PRESERVE_BLOCK
X 1
I 2
PRAGMA PRESERVE_BLOCK
PRAGMA END_PRESERVE_BLOCK
CNOT 1 2
PRAGMA PRESERVE_BLOCK
PRAGMA END_PRESERVE_BLOCK
MEASURE 1 ro[1]
MEASURE 2 ro[2]
"""
        labels = [Label(('Gi', 'Q1')), Label(('Gxpi', 'Q1')), Label('Gcnot', ('Q1', 'Q2'))]
        c = circuit.Circuit(layer_labels=labels, line_labels=['Q1', 'Q2'])
        s = c.convert_to_quil()
        self.assertEqual(quil_str, s)

    def test_done_editing(self):
        self.c.done_editing()
        with self.assertRaises(AssertionError):
            self.c.clear()

    def test_simulate(self):
        # TODO optimize
        # Create a pspec, to test the circuit simulator.
        n = 4
        qubit_labels = ['Q' + str(i) for i in range(n)]
        gate_names = ['Gh', 'Gp', 'Gxpi', 'Gpdag', 'Gcnot']  # 'Gi',
        ps = ProcessorPack(n, gate_names=gate_names, qubit_labels=qubit_labels, construct_models=('target','clifford'))

        # Tests the circuit simulator
        c = circuit.Circuit(layer_labels=[Label('Gh', 'Q0'), Label('Gcnot', ('Q0', 'Q1'))], line_labels=['Q0', 'Q1'])
        out = c.simulate(ps.models['target'])
        self.assertLess(abs(out['00'] - 0.5), 10**-10)
        self.assertLess(abs(out['11'] - 0.5), 10**-10)


class CircuitOperationTester(BaseCase):
    # TODO merge with CircuitMethodTester
    def setUp(self):
        self.s1 = circuit.Circuit(('Gx', 'Gx'), stringrep="Gx^2")
        self.s2 = circuit.Circuit(self.s1, stringrep="Gx^2")

    def test_eq(self):
        self.assertEqual(self.s1, ('Gx', 'Gx'))
        self.assertEqual(self.s2, ('Gx', 'Gx'))
        self.assertTrue(self.s1 == self.s2)

    def test_add(self):
        s3 = self.s1 + self.s2
        self.assertEqual(s3, ('Gx', 'Gx', 'Gx', 'Gx'))

    def test_pow(self):
        s4 = self.s1**3
        self.assertEqual(s4, ('Gx', 'Gx', 'Gx', 'Gx', 'Gx', 'Gx'))

    def test_copy(self):
        s5 = self.s1
        s6 = copy.copy(self.s1)
        s7 = copy.deepcopy(self.s1)

        self.assertEqual(self.s1, s5)
        self.assertEqual(self.s1, s6)
        self.assertEqual(self.s1, s7)

    def test_lt_gt(self):
        self.assertFalse(self.s1 < self.s2)
        self.assertFalse(self.s1 > self.s2)

        s3 = self.s1 + self.s2
        self.assertTrue(self.s1 < s3)
        self.assertTrue(s3 > self.s1)

    def test_read_only(self):
        with self.assertRaises(AssertionError):
            self.s1[0] = 'Gx'  # cannot set items - like a tuple they're read-only

    def test_raise_on_add_non_circuit(self):
        with self.assertRaises(AssertionError):
            self.s1 + ("Gx",)  # can't add non-Circuit to circuit

    def test_clear(self):
        c = self.s1.copy(editable=True)
        self.assertEqual(c.size, 2)
        c.clear()
        self.assertEqual(c.size, 0)


class CompressedCircuitTester(BaseCase):
    def test_compress_op_label(self):
        mdl = circuit.Circuit(None, stringrep="Gx^100")

        comp_init = circuit.CompressedCircuit(mdl, max_period_to_look_for=100)
        pkl_unpkl = pickle.loads(pickle.dumps(comp_init))
        self.assertEqual(comp_init._tup, pkl_unpkl._tup)

        comp_gs = circuit.CompressedCircuit.compress_op_label_tuple(tuple(mdl))
        self.assertEqual(comp_init._tup, comp_gs)

        exp_gs = circuit.CompressedCircuit.expand_op_label_tuple(comp_gs)
        self.assertEqual(tuple(mdl), exp_gs)

    def test_compress_expand(self):
        s1 = circuit.Circuit(('Gx', 'Gx'), stringrep="Gx^2")
        c1 = circuit.CompressedCircuit(s1)
        s1_expanded = c1.expand()
        self.assertEqual(s1, s1_expanded)

    def test_raise_on_construct_from_non_circuit(self):
        with self.assertRaises(ValueError):
            circuit.CompressedCircuit(('Gx',))  # can only create from Circuits
