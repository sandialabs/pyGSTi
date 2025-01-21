import pytest

import pickle

from pygsti.baseobjs.label import Label as L

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
        l = L('Gx', (0, 1), time=1.2, args=('1.4', '1.7'))
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args, ('1.4', '1.7'))
        self.assertEqual(tuple(l), ('Gx', 4, '1.4', '1.7', 0, 1))

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
