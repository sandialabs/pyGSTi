import pytest

import pickle

from pygsti.baseobjs.label import Label as L
from pygsti.baseobjs.label import LabelTup, LabelTupTup, CircuitLabel, LabelStr

from pygsti.serialization import jsoncodec
from pygsti.circuits import Circuit
from ..util import BaseCase

labels = [
    L('Gx', 0),  # a LabelTup
    L('Gx', (0, 1)),  # a LabelTup
    L(('Gx', 0, 1)),  # a LabelTup
    L('Gx'),  # a LabelStr
    L('Gx', None),  # still a LabelStr
    L([('Gx', 0), ('Gy', 0)]),  # a LabelTupTup of LabelTup objs
    L((('Gx', None), ('Gy', None))),  # a LabelTupTup of LabelStr objs
    L([('Gx', 0)]),  # just a LabelTup b/c only one component
    L([L('Gx'), L('Gy')]),  # a LabelTupTup of LabelStrs
    L(L('Gx'))  # Init from another label
]

@pytest.mark.parametrize('label', labels)
def test_to_native(label):
    native = label.to_native()
    from_native = L(native)
    assert label == from_native

@pytest.mark.parametrize('label', labels)
def test_pickle(label):
    s = pickle.dumps(label)
    l2 = pickle.loads(s)
    assert type(label) == type(l2)

@pytest.mark.parametrize('label', labels)
def test_json_encode(label):
    j = jsoncodec.encode_obj(label, False)
    l2 = jsoncodec.decode_obj(j, False)
    assert type(label) == type(l2)

class LabelTester(BaseCase):
    def test_circuit_init(self):
        #Check that parallel operation labels get converted to circuits properly
        opstr = Circuit(((('Gx', 0), ('Gy', 1)), ('Gcnot', 0, 1)))
        c = Circuit(layer_labels=opstr, num_lines=2)
        print(c._labels)
        self.assertEqual(c._labels, (L((('Gx', 0), ('Gy', 1))), L('Gcnot', (0, 1))))

    def test_labels_with_time_and_arguments(self):
        #Label with time and args
        l = L('Gx', (0, 1), time=1.2, args=('1.4', '1.7')) # LabelTupWithArgs
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args, ('1.4', '1.7'))
        self.assertEqual(tuple(l), ('Gx', 4, '1.4', '1.7', 0, 1))
        self.assertEqual(tuple(l + l.name), ('GxGx', 4, '1.4', '1.7', 0, 1))

        l2 = L(('Gx', ';1.4', ';1.7', 0, 1, '!1.25'))
        self.assertEqual(tuple(l2), ('Gx', 4, '1.4', '1.7', 0, 1))

        l3 = L(('Gx', ';', '1.4', ';', '1.7', 0, 1, '!', 1.3))
        self.assertEqual(tuple(l3), ('Gx', 4, '1.4', '1.7', 0, 1))

        self.assertTrue(l == l2 == l3)

        #Time without args
        l = L('Gx', (0, 1), time=1.2)
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args, ())
        self.assertEqual(tuple(l), ('Gx', 0, 1))

        #Args without time
        l = L('Gx', (0, 1), args=('1.4',))
        self.assertEqual(l.time, 0)
        self.assertEqual(l.args, ('1.4',))
        self.assertEqual(tuple(l), ('Gx', 3, '1.4', 0, 1))

    def test_label_time_is_not_hashed(self):
        #Ensure that time is not considered in the equality (or hashing) of labels - it's a
        # tag-along "comment" that does not change the real value of a Label.
        l1 = L('Gx', time=1.2)
        l2 = L('Gx')
        self.assertEqual(l1, l2)
        self.assertTrue(l1.time != l2.time)

        l1 = L('Gx', (0,), time=1.2)
        l2 = L('Gx', (0,))
        self.assertEqual(l1, l2)
        self.assertTrue(l1.time != l2.time)

    def test_only_nonzero_time_is_printed(self):
        l = L('GrotX', (0, 1), args=('1.4',))
        self.assertEqual(str(l), "GrotX;1.4:0:1")  # make sure we don't print time when it's not given (i.e. zero)
        self.assertEqual(l.time, 0.0)  # BUT l.time is still 0, not None
        l = L('GrotX', (0, 1), args=('1.4',), time=0.2)
        self.assertEqual(str(l), "GrotX;1.4:0:1!0.2")  # make sure we do print time when it's nonzero
        self.assertEqual(l.time, 0.2)

    def test_need_to_explicitly_say_its_sorted(self):

        l = L((('Gx', 0),('Ga', 1)))
        self.assertFalse(l.is_sorted, "We will not check for sorting unless asked. The starting assumption is that the label is not sorted.")

        for label in labels:
            self.assertTrue(hasattr(label,"is_sorted"), f"label {label} which has type {type(label)} does not have an attribute to check for sorting.")
            self.assertTrue(hasattr(label,"_is_sorted"), f"label {label} which has type {type(label)} does not have the data member which holds its sorted state.")

            if isinstance(label, (LabelTup, LabelStr)):
                self.assertTrue(label.is_sorted, f"{type(label)} should be sorted by default since there is only one value in it.")
            elif isinstance(label, CircuitLabel):
                self.assertFalse(label.is_sorted, f"{CircuitLabel} can have multiple objects within a single layer so must be checked for sortedness.")
            elif isinstance(label, LabelTupTup):
                self.assertFalse(label.is_sorted, f"{LabelTupTup} can have multiple objects within a single layer so must be checked for sortedness.")

        l = l.with_sorted_inner_labels()
        self.assertTrue(l.is_sorted, f"We just sorted the label {l}!")

    def test_sortedness_respects_order_of_two_qubit_gates(self):
        label1 = L((("Gx", 1), ("GCnot", 0, 2)))
        label2 = L((("Gx", 1), ("GCnot", 2, 0)))
        self.assertNotEqual(label1.with_sorted_inner_labels(), label2.with_sorted_inner_labels())
        # Note that if one uses this for checking that this random circuit has not already been done before
        # then there is a possibility that it could return false even if it has been done before.
        # Consider the case that one assumes a two qubit gate Gii which is a noisy idle gate.
        # If we assume that there is not a handedness to the noisy idle so that Gii 0, 2 == Gii 2, 0.
        # then this sorting check will not handle this.
        # Since a two qubit gate typically has a handedness structure this behavior is desired.

    def test_label_sorting_circuit_labels_recurses_correctly(self):

        label1 = L((("Gx", 1), ("GCnot", 2, 0)))

        cirLabel = CircuitLabel('mycir', [label1,], None, 5, None)

        self.assertEqual(cirLabel.components[0], label1)
        self.assertEqual(cirLabel.with_sorted_inner_labels().components[0], label1.with_sorted_inner_labels())

        cirLabel2 = CircuitLabel('embedded', [label1, cirLabel, L('Gz', 2)], None, 1, None)

        sorted_c2 = cirLabel2.with_sorted_inner_labels()

        for comp in sorted_c2.components:
            self.assertTrue(comp.is_sorted)


    def test_labeltuptup_sorting_recurses_if_necessary(self):

        inner_label1 = L('Gx', 1)
        inner_label2 = L('Gz', 2)
        out_of_order = L((inner_label2, inner_label1))

        in_order = L((inner_label1, inner_label2))

        cirLabelOOO = L((out_of_order,
                         CircuitLabel('embedded', [out_of_order,], (1,2), 2, None).map_state_space_labels({1: 0, 2: 3}),
                         out_of_order.map_state_space_labels({1: 4, 2: 5})))

        cirLabelInOrder = L((CircuitLabel('embedded', [in_order,], (1,2), 2, None).map_state_space_labels({1:0, 2:3}),
                             in_order,
                             in_order.map_state_space_labels({1:4, 2: 5})))
        
        self.assertEqual(cirLabelOOO.with_sorted_inner_labels().components, cirLabelInOrder.components)

    def test_label_rep_evalulation(self):
        """ Make sure Label reps evaluate back to the correct Label """
        from pygsti.baseobjs import Label, CircuitLabel
        labels_to_test = []
        l = Label(('Gx', 0)); labels_to_test.append(l)
        l = Label('Gx', (0,)); labels_to_test.append(l)
        l = Label('rho0'); labels_to_test.append(l)
        l = Label((('Gx', 0), ('Gy', 1))); labels_to_test.append(l)
        l = Label(('Gx', 0, ';', 0.1)); labels_to_test.append(l)
        l = Label('Gx', (0,), args=(0.1,)); labels_to_test.append(l)

        l = Label(('Gx', 0), time=1.0); labels_to_test.append(l)
        l = Label('Gx', (0,), time=1.0); labels_to_test.append(l)
        l = Label('rho0', time=1.0); labels_to_test.append(l)
        l = Label((('Gx', 0), ('Gy', 1)), time=1.0); labels_to_test.append(l)
        l = Label(('Gx', 0, ';', 0.1, '!', 1.0)); labels_to_test.append(l)
        l = Label('Gx', (0,), args=(0.1,), time=1.0); labels_to_test.append(l)

        c = Circuit("[Gx:0Gy:1]^2[Gi]", line_labels=(0,1))
        l = c.to_label(); labels_to_test.append(l)

        for l in labels_to_test:
            print(l, type(l).__name__, end='')
            r = repr(l)
            print(' => eval ' + r)
            evald_repr_l = eval(r)
            self.assertEqual(l, evald_repr_l)
