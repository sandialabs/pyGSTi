
from ..util import BaseCase
from pygsti.circuits import Circuit, CircuitList

class CircuitListTestCase(BaseCase):
    def setUp(self):
        self.cl = CircuitList([Circuit('Gx'), Circuit('Gy')])

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
