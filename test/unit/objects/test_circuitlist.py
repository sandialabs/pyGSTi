
from ..util import BaseCase
from pygsti.circuits import Circuit, CircuitList

class CircuitListTester(BaseCase):
    def setUp(self):
        self.cl = CircuitList([Circuit('Gx'), Circuit('Gy')])

    def test_tensor_circuits_raises_on_circuit(self):
        cl = CircuitList([Circuit('Gx'), Circuit('Gy')])
        c = Circuit('Gz')
        with self.assertRaises(TypeError):
            cl.tensor_circuits(c)

    def test_tensor_circuits_raises_on_diff_length_subcircuits(self):
        cl = CircuitList([Circuit('Gx'), Circuit(['Gy', 'Gz'])])
        c = CircuitList([Circuit('Gz'), Circuit('Gx')])
        #The below should fail as the subcircuits are not the same length.
        with self.assertRaises(ValueError):
            cl.tensor_circuits(c)

    def test_circuitlist_is_unmutable(self):
        with self.assertRaises(TypeError):
            self.cl[0] = Circuit('Gi')

    def test_circuitlist_construction(self):
        #Should not be able to construct a list of lists.
        with self.assertRaises(ValueError):
            cl = CircuitList([CircuitList([Circuit('Gx')])])

    def test_cast(self):
        l = [Circuit('Gx'), Circuit('Gy')]
        cl = CircuitList.cast(l)
        self.assertIsInstance(cl, CircuitList)

    def test_apply_aliases(self):
        aliased_list = CircuitList([Circuit('Gx', name="test_name")], op_label_aliases={'Gx': Circuit('Gy')})
        self.assertEqual(aliased_list.apply_aliases()[0], Circuit('Gy'))
    
    def test_truncate(self):
        truncated_list = self.cl.truncate([Circuit('Gx')])
        self.assertEqual(len(truncated_list), 1)
        self.assertEqual(truncated_list[0], Circuit('Gx'))

    def test_truncate_to_dataset(self):
        from pygsti.data import DataSet
        ds = DataSet(outcome_labels=['0', '1'])
        ds.add_count_dict(Circuit('Gx'), {'0': 10, '1': 90})
        truncated_list = self.cl.truncate_to_dataset(ds)
        self.assertEqual(len(truncated_list), 1)
        self.assertEqual(truncated_list[0], Circuit('Gx'))

    def test_tensor_circuits(self):
        cl1 = CircuitList([Circuit('Gx', line_labels=[0]), Circuit('Gy', line_labels=[0])])
        cl2 = CircuitList([Circuit('Gz', line_labels=[1]), Circuit('Ga', line_labels=[1])])
        tensored_list = cl1.tensor_circuits(cl2)
        self.assertEqual(len(tensored_list), 2)
        c1 = Circuit([(('Gx', 0), ('Gz', 1))])
        c2 = Circuit([(('Gy', 0), ('Ga', 1))])
        self.assertEqual(tensored_list[0], c1)
        self.assertEqual(tensored_list[1], c2)

    def test_tensor_circuits_raises_value_error_on_self_tensor(self):
        cl1 = CircuitList([Circuit('Gx:0'), Circuit('Gy:0')])
        with self.assertRaises(ValueError):
            cl1.tensor_circuits(cl1)

    def test_tensor_circuits_str_rep(self):
        cl1 = CircuitList([Circuit('Gx:0'), Circuit('Gy:0')])
        cl2 = CircuitList([Circuit('Gz:1'), Circuit('Ga:1')])
        tensored_list = cl1.tensor_circuits(cl2)
        self.assertEqual(len(tensored_list), 2)
        c1 = Circuit([(('Gx', 0), ('Gz', 1))])
        c2 = Circuit([(('Gy', 0), ('Ga', 1))])
        self.assertEqual(tensored_list[0], c1)
        self.assertEqual(tensored_list[1], c2)

        tensored_reversed_list = cl2.tensor_circuits(cl1)
        self.assertEqual(len(tensored_reversed_list), 2)
        c1 = Circuit([(('Gz', 1), ('Gx', 0))], line_labels=(1,0))
        c2 = Circuit([(('Ga', 1), ('Gy', 0))], line_labels=(1,0))
        self.assertEqual(tensored_reversed_list[0], c1)
        self.assertEqual(tensored_reversed_list[1], c2)

    def test_tensor_circuits_numeric_lines(self):

        cl1 = CircuitList([Circuit([('Gx', 0)]), Circuit([('Gy', 0)])])
        cl2 = CircuitList([Circuit([('Gz', 1)]), Circuit([('Ga', 1)])])
        tensored_list = cl1.tensor_circuits(cl2)
        self.assertEqual(len(tensored_list), 2)
        c1 = Circuit([(('Gx', 0), ('Gz', 1))])
        c2 = Circuit([(('Gy', 0), ('Ga', 1))])
        self.assertEqual(tensored_list[0], c1)
        self.assertEqual(tensored_list[1], c2)

        tensored_reversed_list = cl2.tensor_circuits(cl1)
        self.assertEqual(len(tensored_reversed_list), 2)
        c1 = Circuit([(('Gz', 1), ('Gx', 0))], line_labels=(1,0))
        c2 = Circuit([(('Ga', 1), ('Gy', 0))], line_labels=(1,0))
        self.assertEqual(tensored_reversed_list[0], c1)
        self.assertEqual(tensored_reversed_list[1], c2)

    def test_tensor_product_of_every_circuit_construction_form(self):
        from pygsti.baseobjs.label import Label
        # Every way to construct a circuit that is equivalent to Circuit([("Gx", 0)])
        c1_options = [
            Circuit([('Gx', 0)]),
            Circuit([(('Gx', 0))]),
            Circuit('Gx:0'),
            Circuit([Label('Gx', 0)]),
            Circuit(layer_labels=[('Gx',0)])
        ]

        # Every way to construct a circuit that is equivalent to Circuit([("Gy", 1)])
        c2_options = [
            Circuit([('Gy', 1)]),
            Circuit([(('Gy', 1))]),
            Circuit('Gy:1'),
            Circuit([Label('Gy', 1)]),
            Circuit(layer_labels=[('Gy',1)])
        ]

        expected_revc = Circuit([(("Gx", 0), ("Gy",1))], line_labels=(1,0))
        expected_c = Circuit([(('Gx', 0), ('Gy', 1))])
        for i1, c1 in enumerate(c1_options):
            for i2, c2 in enumerate(c2_options):
                cl1 = CircuitList([c1])
                cl2 = CircuitList([c2])
                tensored_list = cl1.tensor_circuits(cl2)
                self.assertEqual(len(tensored_list), 1)
                self.assertEqual(tensored_list[0], expected_c, f"Testing options {i1} and {i2}")

        for i1, c1 in enumerate(c1_options):
            for i2, c2 in enumerate(c2_options):
                tensored_reverse_list = cl2.tensor_circuits(cl1)
                self.assertEqual(len(tensored_reverse_list), 1)
                self.assertEqual(tensored_reverse_list[0], expected_revc, f"Testing options {i1} and {i2}")

        # The following combinations are expected to error on account of ambiguity.
        c2_bad_options = [
            Circuit('Gy@1')
        ]
        c1_bad_options = [
            Circuit('Gx@0')
        ]
        for i1, c1 in enumerate(c1_options):
            for i2, c2 in enumerate(c2_bad_options):
                cl1 = CircuitList([c1])
                cl2 = CircuitList([c2])
                with self.assertRaises(ValueError):
                    cl1.tensor_circuits(cl2)
                with self.assertRaises(ValueError):
                    cl2.tensor_circuits(cl1)

        for i1, c1 in enumerate(c1_bad_options):
            for i2, c2 in enumerate(c2_options):
                cl1 = CircuitList([c1])
                cl2 = CircuitList([c2])
                with self.assertRaises(ValueError):
                    cl1.tensor_circuits(cl2)
                with self.assertRaises(ValueError):
                    cl2.tensor_circuits(cl1)