import pytest

import pickle

from pygsti.baseobjs.label import Label as L, LabelTup, LabelTupTup, CircuitLabel, LabelStr, LabelTupTupWithTime, LabelTupWithArgs, LabelTupTupWithArgs, LabelTupWithTime

from pygsti.serialization import jsoncodec
from pygsti.circuits import Circuit
from ..util import BaseCase


labels = [
    L('Gx', 0),  # a LabelTup
    L('Gx', (0, 1)),  # a LabelTup
    L(('Gx', 0, 1)),  # a LabelTup
    L('Gx'),  # a LabelStr
    L('Gx', None),  # still a LabelStr
    LabelStr.init('rho0', 1.5),  # LabelStr objects have a `time` field
    L([('Gx', 0), ('Gy', 0)]),  # a LabelTupTup of LabelTup objs
    L((('Gx', None), ('Gy', None))),  # a LabelTupTup of LabelStr objs
    L([('Gx', 0)]),  # just a LabelTup b/c only one component
    L([L('Gx'), L('Gy')]),  # a LabelTupTup of LabelStrs
    L(L('Gx')),  # Init from another label
    CircuitLabel('circuit', [("Gx", 1), ("Gz", 2)], None, 1, None),
    L(('Gx', 0), time=0.1),  # LabelTupWithTime
    L(('Gx', 0), time=0.1, args=("foo",)),  # LabelTupWithArgs
    L([("Gx", 0), ("Gy", 1)], time=3.1),  # LabelTupTupWithTime
    L([("Gx", 0), ("Gy", 1)], time=0.0459, args=("bar",)),  # LabelTupTupWithArgs
    CircuitLabel('', (L('Gx', (0,)),), (0,), 1, time=3.0), # CircuitLabel with nonzero time
    CircuitLabel('', (L('Gx', (0,)),), (0,), 1, time=0.0) # CircuitLabel with time
]


@pytest.mark.parametrize('label', labels)
def test_to_native(label):
    native = label.to_native()
    from_native = L(native)
    assert label == from_native
    assert isinstance(label, type(from_native))
    assert isinstance(from_native, type(label))


@pytest.mark.parametrize('label', labels)
def test_pickle(label):
    s = pickle.dumps(label)
    l2 = pickle.loads(s)
    assert isinstance(label, type(l2))
    assert isinstance(l2, type(label))
    assert label == l2


@pytest.mark.parametrize('label', labels)
def test_json_encode(label):
    j = jsoncodec.encode_obj(label, False)
    l2 = jsoncodec.decode_obj(j, False)
    assert isinstance(label, type(l2))
    assert isinstance(l2, type(label))
    assert label == l2


@pytest.mark.parametrize('label', labels)
def test_hashable(label):
    # __hash__ is needed for insertion into a set
    s = {label}
    assert label in s


def test_legacy_to_native_deserialization():
    # 1. LabelTup
    assert L(('Gx', 0)) == L('Gx', 0)
    assert isinstance(L(('Gx', 0)), LabelTup)

    # 2. LabelStr
    assert L('Gx') == L('Gx')
    assert isinstance(L('Gx'), LabelStr)

    # 3. LabelTupTup
    assert L((('Gx', 0), ('Gy', 1))) == L([('Gx', 0), ('Gy', 1)])
    assert isinstance(L((('Gx', 0), ('Gy', 1))), LabelTupTup)

    # 4. CircuitLabel
    cl = L(('c', None, 1, ('Gx', 1), ('Gz', 2)))
    assert cl == CircuitLabel('c', [('Gx', 1), ('Gz', 2)], None, 1, None)
    assert isinstance(cl, CircuitLabel)

    # 5. LabelTupWithArgs (Legacy raw: K=3) - non-homogeneous -> args detected
    lwa = L(('Gx', 3, '1.4', 0))
    assert lwa == L('Gx', 0, args=('1.4',))
    assert isinstance(lwa, LabelTupWithArgs)

    # 6. LabelTupTupWithArgs (Legacy raw: K=2)
    lttwa = L((2, 'foo', ('Gx', 0), ('Gy', 1)))
    assert lttwa == L([('Gx', 0), ('Gy', 1)], args=('foo',))
    assert isinstance(lttwa, LabelTupTupWithArgs)


class LabelTester(BaseCase):

    def test_circuit_init(self):
        #Check that parallel operation labels get converted to circuits properly
        opstr = Circuit(((('Gx', 0), ('Gy', 1)), ('Gcnot', 0, 1)))
        c = Circuit(layer_labels=opstr, num_lines=2)
        print(c._labels)
        self.assertEqual(c._labels, (L((('Gx', 0), ('Gy', 1))), L('Gcnot', (0, 1))))

    def test_labels_with_time_and_arguments(self):
        #Label with time and args
        l = L('Gx', (0, 1), time=1.2, args=('1.4', '1.7'))  # LabelTupWithArgs
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
        self.assertEqual(l1.time, 1.2)
        self.assertNotEqual(l1.time, l2.time)  # LabelStr can still have a time attribute if parsed.

        l1 = L('Gx', (0,), time=1.2)
        l2 = L('Gx', (0,))
        self.assertEqual(l1, l2)
        self.assertEqual(l1.time, 1.2)
        self.assertTrue(not hasattr(l2, "time"))

    def test_only_nonzero_time_is_printed(self):
        l = L('GrotX', (0, 1), args=('1.4',))
        self.assertEqual(str(l), "GrotX;1.4:0:1")  # make sure we don't print time when it's not given (i.e. zero)
        self.assertEqual(l.time, 0.0)  # BUT l.time is still 0, not None
        l = L('GrotX', (0, 1), args=('1.4',), time=0.2)
        self.assertEqual(str(l), "GrotX;1.4:0:1!0.2")  # make sure we do print time when it's nonzero
        self.assertEqual(l.time, 0.2)

    def test_empty_label_with_time(self):
        l = L((), time=1.2)
        self.assertIsInstance(l, LabelTupTupWithTime)
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.sslbls, None)
        self.assertEqual(l.qubits, None)
        self.assertEqual(l.num_qubits, None)
        self.assertEqual(l.name, 'COMPOUND')

    def test_label_with_none_sslbls_and_args(self):
        l = L('Gx', None, args=('foo',))
        self.assertIsInstance(l, LabelTupWithArgs)
        self.assertEqual(l.args, ('foo',))
        self.assertEqual(l.sslbls, None)
        self.assertEqual(l.qubits, None)
        self.assertEqual(l.num_qubits, None)
        self.assertEqual(l.name, 'Gx')

    def test_label_reps_property(self):
        l = L('Gx')
        self.assertEqual(l.reps, 1)
        self.assertEqual(l.name, 'Gx')

    def test_label_strip_args(self):
        l = L('Gx', args=('foo',))
        l2 = l.strip_args()
        self.assertEqual(l2, L('Gx'))
        self.assertEqual(l2.name, 'Gx')

    def test_label_is_simple(self):
        l = L('Gx')
        self.assertTrue(l.is_simple)
        self.assertEqual(l.name, 'Gx')

    def test_label_with_sorted_inner_labels(self):
        l = L((('Gx', 1), ('Gy', 0)))
        sorted_l = l.with_sorted_inner_labels()
        self.assertEqual(sorted_l, L((('Gy', 0), ('Gx', 1))))
        self.assertEqual(sorted_l.name, 'COMPOUND')
        self.assertEqual(sorted_l.sslbls, (0, 1))
        self.assertEqual(sorted_l.qubits, (0, 1))
        self.assertEqual(sorted_l.num_qubits, 2)

    def test_with_sorted_inner_labels_idempotent(self):
        # Once a label is sorted, calling with_sorted_inner_labels again should
        # short-circuit on the `is_sorted` check and return the same object.
        l = L((('Gy', 0), ('Gx', 1)))
        sorted_l = l.with_sorted_inner_labels()
        self.assertTrue(sorted_l.is_sorted)
        self.assertIs(sorted_l.with_sorted_inner_labels(), sorted_l)

    def test_with_sorted_inner_labels_none_sslbls_returns_self(self):
        # If any inner label has sslbls == None we cannot sort, so `self` is
        # returned unchanged.
        l = L((('Gx', None), ('Gy', None)))
        self.assertIsInstance(l, LabelTupTup)
        self.assertFalse(l.is_sorted)
        self.assertIs(l.with_sorted_inner_labels(), l)

        l = L((('Gx', None), ('Gy', 2)))
        self.assertIsInstance(l, LabelTupTup)
        self.assertFalse(l.is_sorted)
        self.assertIs(l.with_sorted_inner_labels(), l)

        l = L((('Gx', 1), ('Gy', None)))
        self.assertIsInstance(l, LabelTupTup)
        self.assertFalse(l.is_sorted)
        self.assertIs(l.with_sorted_inner_labels(), l)

    def test_unparsable_string_falls_back_to_labelstr(self):
        # A string containing parse-significant characters (':', '!', ';') that
        # fails to parse should fall back to being treated as a plain LabelStr.
        l = L('not a real::label!!')
        self.assertIsInstance(l, LabelStr)
        self.assertEqual(str(l), 'not a real::label!!')

    def test_labeltuptup_args_not_propagated(self):
        import warnings
        l1 = L('Gzr', ('Q0',), args=('1.025087886927528',))
        l2 = L('Gzr', ('Q1',), args=('1.3957662527842025',))

        # Create a compound label from components with arguments. Building a
        # plain (timeless) LabelTupTup out of args-bearing components whose
        # times defaulted to 0.0 is a routine, unambiguous operation and must
        # not emit a metadata hint.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compound = L((l1, l2))
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    self.assertNotIn("default time to 0.0", str(warning.message))

        # Check that the compound label itself has no arguments
        self.assertEqual(compound.args, ())

        # Check that the string representation is clean
        self.assertEqual(
            repr(compound),
            "Label((Label('Gzr',('Q0',),args=('1.025087886927528',)), Label('Gzr',('Q1',),args=('1.3957662527842025',))))"
        )

    def test_label_copy(self):
        l = L('Gx')
        l_copy = l.copy()
        self.assertEqual(l, l_copy)
        self.assertIsNot(l, l_copy)
        self.assertEqual(l_copy.name, 'Gx')

    def test_label_replace_name(self):
        l = L('Gx')
        l2 = l.replace_name('Gx', 'Gy')
        self.assertEqual(l2, L('Gy'))
        self.assertEqual(l2.name, 'Gy')

    def test_need_to_explicitly_say_its_sorted(self):

        l = L((('Gx', 0), ('Ga', 1)))
        self.assertFalse(l.is_sorted, "We will not check for sorting unless asked. The starting assumption is that the label is not sorted.")

        for label in labels:
            self.assertTrue(hasattr(label, "is_sorted"), f"label {label} which has type {type(label)} does not have an attribute to check for sorting.")

            if isinstance(label, (LabelTup, LabelStr)):
                self.assertTrue(label.is_sorted, f"{type(label)} should be sorted by default since there is only one value in it.")
            elif isinstance(label, CircuitLabel):
                self.assertFalse(label.is_sorted, f"{CircuitLabel} can have multiple objects within a single layer so must be checked for sortedness.")
                self.assertTrue(hasattr(label, "_is_sorted"), f"label {label} which has type {type(label)} does not have the data member which holds its sorted state.")
            elif isinstance(label, LabelTupTup):
                self.assertFalse(label.is_sorted, f"{LabelTupTup} can have multiple objects within a single layer so must be checked for sortedness.")
                self.assertTrue(hasattr(label, "_is_sorted"), f"label {label} which has type {type(label)} does not have the data member which holds its sorted state.")

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
                         CircuitLabel('embedded', [out_of_order,], (1, 2), 2, None).map_state_space_labels({1: 0, 2: 3}),
                         out_of_order.map_state_space_labels({1: 4, 2: 5})))

        cirLabelInOrder = L((CircuitLabel('embedded', [in_order,], (1, 2), 2, None).map_state_space_labels({1: 0, 2: 3}),
                             in_order,
                             in_order.map_state_space_labels({1: 4, 2: 5})))

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

        c = Circuit("[Gx:0Gy:1]^2[Gi]", line_labels=(0, 1))
        l = c.to_label(); labels_to_test.append(l)

        for l in labels_to_test:
            print(l, type(l).__name__, end='')
            r = repr(l)
            print(' => eval ' + r)
            evald_repr_l = eval(r)
            self.assertEqual(l, evald_repr_l)


class LabelStrTester(BaseCase):

    def test_labelstr_add(self):
        l1 = L('Gx')
        l2 = l1 + '_foo'
        self.assertEqual(l2, L('Gx_foo'))
        self.assertEqual(l2.name, 'Gx_foo')

        with self.assertRaises(NotImplementedError):
            l1 + L(("Gx", 0))

        lbl = L('foo', ())
        lbl = lbl + "_bar"
        # there should not be an error when adding to a null statespace LabelStr.
        self.assertEqual(lbl.name, "foo_bar")

    def test_labelstr_add_labelstr(self):
        l1 = L('Gx')
        l2 = L('Gy')
        l3 = l1 + l2
        self.assertEqual(l3, L('GxGy'))
        self.assertEqual(l3.name, 'GxGy')

    def test_labelstr_replace_name_no_match(self):
        l = L('Gx')
        l2 = l.replace_name('Gy', 'Gz')
        self.assertEqual(l2, L('Gx'))
        self.assertEqual(l2.name, 'Gx')

    def test_repr_and_str_differ(self):
        l = L(('Gx', 0))
        self.assertNotEqual(repr(l), str(l))

    def test_labelstr_expand_subcircuits(self):
        l = L('Gx')
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labelstr_has_prefix(self):
        l = L('Gx_foo')
        self.assertTrue(l.has_prefix('Gx'))
        self.assertFalse(l.has_prefix('Gy'))

    def test_labelstr_strip_args(self):
        l = L('Gx_foo')
        self.assertIsInstance(l, LabelStr)
        l2 = l.strip_args()
        self.assertEqual(l2, L('Gx_foo'))

    def test_labelstr_lt_gt(self):
        l1 = L('A')
        l2 = L('B')
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)

    def test_labelstr_to_native(self):
        l = L('Gx')
        self.assertEqual(l.to_native(), 'Gx')

    def test_labelstr_with_time_serialization_roundtrip(self):
        # Regression: a timed LabelStr must round-trip with its name AND time
        # intact through pickle and the JSON codec.  The parametrized round-trip
        # tests only assert ==, which ignores time, so we check metadata explicitly.
        import pickle
        l = LabelStr.init('rho0', 1.5)
        self.assertEqual(l.name, 'rho0')
        self.assertEqual(l.time, 1.5)

        # pickle
        pl = pickle.loads(pickle.dumps(l))
        self.assertIsInstance(pl, LabelStr)
        self.assertEqual(pl.name, 'rho0')
        self.assertEqual(pl.time, 1.5)

        # JSON codec
        jl = jsoncodec.decode_obj(jsoncodec.encode_obj(l, False), False)
        self.assertIsInstance(jl, LabelStr)
        self.assertEqual(jl.name, 'rho0')
        self.assertEqual(jl.time, 1.5)

    def test_labelstr_contains(self):
        l = L('Gx_foo')
        self.assertTrue('Gx' in l)
        self.assertFalse('Gy' in l)

    def test_labelstr_properties(self):

        l = L("Gxfoo")
        self.assertEqual(len(l.args), 0)
        self.assertEqual(l.components, (l,))
        # This information may be stored within the string
        # but that means parsing the string which a LayerRule or Model may handle.
        self.assertEqual(l.qubits, None)
        self.assertEqual(l.num_qubits, None)

    def test_labelstr_gets_parsed(self):

        l = L("Gx;foo")
        self.assertIsInstance(l, LabelTupWithArgs)
        l = L("Gx!0.1")
        self.assertIsInstance(l, LabelStr)
        l = L("Gx:0@0")
        self.assertIsInstance(l, LabelTup)


class LabelTupTester(BaseCase):

    def test_labeltup_replace_name(self):
        l = L(('Gx', 0))
        l2 = l.replace_name('Gx', 'Gy')
        self.assertEqual(l2, L(('Gy', 0)))
        self.assertEqual(l2.name, 'Gy')
        self.assertEqual(l2.sslbls, (0,))
        self.assertEqual(l2.qubits, (0,))
        self.assertEqual(l2.num_qubits, 1)

    def test_labeltup_map_state_space_labels(self):
        l = L(('Gx', 0, 1))
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, L(('Gx', 1, 2)))
        self.assertEqual(l2.name, 'Gx')
        self.assertEqual(l2.sslbls, (1, 2))
        self.assertEqual(l2.qubits, (1, 2))
        self.assertEqual(l2.num_qubits, 2)

    def test_labeltup_add(self):
        l1 = L(('Gx', 0))
        l2 = l1 + '_foo'
        self.assertEqual(l2, L(('Gx_foo', 0)))
        self.assertEqual(l2.name, 'Gx_foo')
        self.assertEqual(l2.sslbls, (0,))

        l = L(("Gx", 0))
        l2 = l + "foo"
        self.assertEqual(l2, L(("Gxfoo", 0)))
        with self.assertRaises(NotImplementedError):
            l + l  # type: ignore

        # Test addition with null state space labels
        lbl = LabelTup.init("foo", ())
        lbl = lbl + "_bar"
        self.assertEqual(lbl.name, "foo_bar")
        return

    def test_labeltup_homogeneity_assert(self):
        # homogeneous is allowed
        LabelTup.init('Gx', (0, 1))
        LabelTup.init('Gx', ('Q0', 'Q1'))
        # heterogeneous raises AssertionError
        with self.assertRaises(AssertionError):
            LabelTup.init('Gx', (0, 'Q1'))

    def test_labeltup_expand_subcircuits(self):
        l = L(('Gx', 0))
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labeltup_has_prefix(self):
        l = L(('Gx_foo', 0))
        self.assertTrue(l.has_prefix('Gx'))
        self.assertFalse(l.has_prefix('Gy'))

    def test_labeltup_strip_args(self):
        l = L(('Gx', 0), args=('foo',))
        l2 = l.strip_args()
        self.assertEqual(l2, L(('Gx', 0)))

    def test_labeltup_lt_gt(self):
        l1 = L(('A', 0))
        l2 = L(('B', 0))
        l3 = L(('A', 1))
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltup_to_native(self):
        l = L(('Gx', 0))
        self.assertEqual(l.to_native(), ('Gx', 0))

    def test_labeltup_contains(self):
        l = L(('Gx', 0))
        self.assertTrue('Gx' in l)
        self.assertTrue(0 in l)
        self.assertFalse('Gy' in l)

    def test_labeltup_canhavenullsslbls(self):
        l = LabelTup.init("foo", ())
        self.assertEqual(l.sslbls, None)


class LabelTupTupTester(BaseCase):

    def test_labeltuptup_replace_name(self):
        l = L((('Gx', 0), ('Gy', 1)))
        l2 = l.replace_name('Gx', 'Gz')
        self.assertEqual(l2, L((('Gz', 0), ('Gy', 1))))
        self.assertEqual(l2.name, 'COMPOUND')
        self.assertEqual(l2.sslbls, (0, 1))
        self.assertEqual(l2.qubits, (0, 1))
        self.assertEqual(l2.num_qubits, 2)

    def test_labeltuptup_map_state_space_labels(self):
        l = L((('Gx', 0), ('Gy', 1)))
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, L((('Gx', 1), ('Gy', 2))))
        self.assertEqual(l2.name, 'COMPOUND')
        self.assertEqual(l2.sslbls, (1, 2))
        self.assertEqual(l2.qubits, (1, 2))
        self.assertEqual(l2.num_qubits, 2)

    def test_labeltuptup_add(self):
        l1 = L((('Gx', 0), ('Gy', 1)))
        with self.assertRaises(NotImplementedError):
            l1 + '_foo'

    def test_labeltuptup_has_prefix(self):
        l = L((('Gx_foo', 0), ('Gy_bar', 1)))
        self.assertTrue(l.has_prefix('G'))
        self.assertFalse(l.has_prefix('Gz'))
        self.assertTrue(l.has_prefix('Gx', typ='any'))
        self.assertFalse(l.has_prefix('Gz', typ='any'))

        with self.assertRaises(ValueError):
            l.has_prefix("Gx", typ="foo")

    def test_labeltuptup_lt_gt(self):
        l1 = L((('A', 0), ('B', 1)))
        l2 = L((('A', 0), ('C', 1)))
        l3 = L((('B', 0), ('A', 1)))
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltuptup_to_native(self):
        l = L((('Gx', 0), ('Gy', 1)))
        self.assertEqual(l.to_native(), (None, None, (('Gx', 0), ('Gy', 1))))

    def test_labeltuptup_contains(self):
        l = L((('Gx', 0), ('Gy', 1)))
        self.assertTrue(L(('Gx', 0)) in l)
        self.assertFalse(L(('Gz', 0)) in l)

    def test_labeltuptup_init_preserves_component_metadata(self):
        l1 = L('Gx', (0,), time=0.1)
        l2 = L('Gy', (1,), args=('foo',))

        # Create a LabelTupTup from components with metadata
        # Note: Label() factory should upgrade this to a LabelTupTupWithArgs
        l_tup_tup = L((l1, l2))

        # Check that the components within the new label still have their metadata
        self.assertEqual(l_tup_tup.components[0].time, 0.1)
        self.assertEqual(l_tup_tup.components[1].args, ('foo',))

        ltt = LabelTupTup.init((l1, l2))
        # Also check with the init methods directly
        self.assertEqual(ltt.components[0].time, 0.1)
        self.assertEqual(ltt.components[1].args, ('foo',))

        # Check that component args are collected properly
        self.assertEqual(ltt.collect_args(), ('foo',))

    def test_with_sorted_inner_labels_raises_on_duplicates_tuptup(self):
        l = LabelTupTup.init((('Gx', 0), ('Gy', 0)))
        with self.assertRaises(ValueError):
            l.with_sorted_inner_labels()

    def test_labeltuptup_warns_on_same_time(self):
        l1 = L('Gx', (0,), time=0.1)
        l2 = L('Gy', (1,), time=0.1)
        with self.assertWarns(RuntimeWarning):
            L((l1, l2))

    def test_labeltuptup_warns_on_same_time_zero(self):
        import warnings
        l1 = LabelTupWithTime.init('Gx', (0,), time=0.0)
        l2 = LabelTupWithTime.init('Gy', (1,), time=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            L((l1, l2))
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    self.assertNotIn("You may want to consider a LabelTupTupWithTime", str(warning.message))


class LabelTupTupWithTimeTester(BaseCase):

    def test_labeltuptupwithtime(self):
        l = LabelTupTupWithTime.init((('Gx', 0), ('Gy', 1)), time=0.1)
        self.assertEqual(l.time, 0.1)
        self.assertEqual(l.to_native(), (0.1, None, (('Gx', 0), ('Gy', 1))))
        self.assertEqual(l.name, 'COMPOUND')
        self.assertEqual(l.sslbls, (0, 1))
        self.assertEqual(l.qubits, (0, 1))
        self.assertEqual(l.num_qubits, 2)
        self.assertEqual(l.components, (L(('Gx', 0)), L(('Gy', 1))))

    def test_labeltuptup_expand_subcircuits(self):
        l = L((('Gx', 0), ('Gy', 1)))
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labeltuptupwithtime_replace_name(self):
        l = LabelTupTupWithTime.init((('Gx', 0), ('Gy', 1)), time=0.1)
        l2 = l.replace_name('Gx', 'Gz')
        self.assertEqual(l2, LabelTupTupWithTime.init((('Gz', 0), ('Gy', 1)), time=0.1))

    def test_labeltuptupwithtime_lt_gt(self):
        l1 = LabelTupTupWithTime.init((('A', 0), ('B', 1)), time=0.1)
        l2 = LabelTupTupWithTime.init((('A', 0), ('C', 1)), time=0.1)
        l3 = LabelTupTupWithTime.init((('B', 0), ('A', 1)), time=0.1)
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltuptupwithtime_map_state_space_labels(self):
        l = LabelTupTupWithTime.init((('Gx', 0), ('Gy', 1)), time=0.1)
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, LabelTupTupWithTime.init((('Gx', 1), ('Gy', 2)), time=0.1))

    def test_labeltuptupwithtime_replace_name_no_match(self):
        l = LabelTupTupWithTime.init((('Gx', 0), ('Gy', 1)), time=0.1)
        l2 = l.replace_name('Gz', 'Ga')
        self.assertEqual(l, l2)

    def test_labeltuptupwithtime_expand_subcircuits(self):
        l = LabelTupTupWithTime.init((('Gx', 0), ('Gy', 1)), time=0.1)
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labeltuptupwithtime_reinit_preserves_toplevel_metadata(self):
        # Test LabelTupTupWithTime
        l_with_time1 = LabelTupTupWithTime.init((('Gx', 0),), time=0.5)
        l_with_time2 = LabelTupTupWithTime.init(l_with_time1)
        self.assertEqual(l_with_time1.time, 0.5)
        self.assertEqual(l_with_time2.time, 0.5)
        self.assertEqual(l_with_time1, l_with_time2)


class LabelTupTupWithArgsTester(BaseCase):

    def test_labeltuptupwithargs(self):
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=('foo',))
        self.assertEqual(l.args, ('foo',))
        self.assertEqual(l.to_native(), (0.0, ('foo',), (('Gx', 0), ('Gy', 1))))
        self.assertEqual(l.name, 'COMPOUND')
        self.assertEqual(l.sslbls, (0, 1))
        self.assertEqual(l.qubits, (0, 1))
        self.assertEqual(l.num_qubits, 2)
        self.assertEqual(l.components, (L(('Gx', 0)), L(('Gy', 1))))

    def test_labeltuptupwithargs_replace_name(self):
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=('foo',))
        l2 = l.replace_name('Gx', 'Gz')
        self.assertEqual(l2, LabelTupTupWithArgs.init((('Gz', 0), ('Gy', 1)), args=('foo',)))

    def test_labeltuptupwithargs_lt_gt(self):
        l1 = LabelTupTupWithArgs.init((('A', 0), ('B', 1)), args=('foo',))
        l2 = LabelTupTupWithArgs.init((('A', 0), ('C', 1)), args=('foo',))
        l3 = LabelTupTupWithArgs.init((('B', 0), ('A', 1)), args=('foo',))
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltuptupwithargs_map_state_space_labels(self):
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=(1,))
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, LabelTupTupWithArgs.init((('Gx', 1), ('Gy', 2)), args=(1,)))
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=(1,))
        l2 = l.map_state_space_labels({0: 1, 1: 2})
        self.assertEqual(l2, LabelTupTupWithArgs.init((('Gx', 1), ('Gy', 2)), args=(1,)))

    def test_labeltuptupwithargs_replace_name_no_match(self):
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=('foo',))
        l2 = l.replace_name('Gz', 'Ga')
        self.assertEqual(l, l2)

    def test_labeltuptupwithargs_expand_subcircuits(self):
        l = LabelTupTupWithArgs.init((('Gx', 0), ('Gy', 1)), args=('foo',))
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labeltuptupwithargs_reinit_preserves_metadata(self):
        # Test LabelTupTupWithArgs
        l_with_args1 = LabelTupTupWithArgs.init((('Gx', 0),), time=0.2, args=('foo',))
        l_with_args2 = LabelTupTupWithArgs.init(l_with_args1)
        self.assertEqual(l_with_args1.time, 0.2)
        self.assertEqual(l_with_args1.args, ('foo',))
        self.assertEqual(l_with_args2.time, 0.2)
        self.assertEqual(l_with_args2.args, ('foo',))
        self.assertEqual(l_with_args1, l_with_args2)

    def test_labeltuptupwithargs_init_preserves_component_metadata(self):
        l1 = L('Gx', (0,), time=0.1)
        l2 = L('Gy', (1,), args=('foo',))
        ltt_wa = LabelTupTupWithArgs.init((l1, l2), args=('bar',))
        self.assertEqual(ltt_wa.components[0].time, 0.1)
        self.assertEqual(ltt_wa.components[1].args, ('foo',))
        self.assertEqual(ltt_wa.collect_args(), ('bar', 'foo'))

        ltt_wa_time = LabelTupTupWithArgs.init((l1, l2), time=5.0, args=("bar",))
        self.assertEqual(ltt_wa_time.components[0].time, 0.1)
        self.assertEqual(ltt_wa_time.components[1].args, ('foo',))
        self.assertEqual(ltt_wa_time.collect_args(), ('bar', 'foo'))
        self.assertEqual(ltt_wa_time.time, 5.0)

        ltt_just_time = LabelTupTupWithArgs.init((l1, l2), time=5.0)
        self.assertEqual(ltt_just_time.components[0].time, 0.1)
        self.assertEqual(ltt_just_time.components[1].args, ('foo',))
        self.assertEqual(ltt_just_time.collect_args(), ('foo',))
        self.assertEqual(ltt_just_time.time, 5.0)
        self.assertIsInstance(ltt_just_time, LabelTupTupWithArgs)  # only because we directly call it.

    def test_labeltuptupwithargs_add(self):
        l1 = LabelTupTupWithArgs.init((('A', 0), ('B', 1)), args=('foo',))
        l2 = LabelTupTupWithArgs.init((('A', 0), ('C', 1)), args=('foo',))
        with self.assertRaises(NotImplementedError):
            l1 + l2


class LabelTupWithTimeTester(BaseCase):

    def test_labeltupwithtime(self):
        l = LabelTupWithTime.init('Gx', (0,), time=0.1)
        self.assertEqual(l.time, 0.1)
        self.assertEqual(l.to_native(), ('Gx', 0, '!0.1'))
        self.assertEqual(l.name, 'Gx')
        self.assertEqual(l.sslbls, (0,))
        self.assertEqual(l.qubits, (0,))
        self.assertEqual(l.num_qubits, 1)
        self.assertEqual(l.components, (l,))

    def test_labeltupwithtime_replace_name(self):
        l = LabelTupWithTime.init('Gx', (0,), time=0.1)
        l2 = l.replace_name('Gx', 'Gy')
        self.assertEqual(l2, LabelTupWithTime.init('Gy', (0,), time=0.1))

    def test_labeltupwithtime_lt_gt(self):
        l1 = LabelTupWithTime.init('A', (0,), time=0.1)
        l2 = LabelTupWithTime.init('B', (0,), time=0.1)
        l3 = LabelTupWithTime.init('A', (1,), time=0.1)
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltupwithtime_map_state_space_labels(self):
        l = LabelTupWithTime.init('Gx', (0, 1), time=0.1)
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, LabelTupWithTime.init('Gx', (1, 2), time=0.1))

        l2 = l.map_state_space_labels({0: 1, 1: 2})
        self.assertEqual(l2, LabelTupWithTime.init('Gx', (1, 2), time=0.1))

    def test_labeltupwithtime_replace_name_no_match(self):
        l = LabelTupWithTime.init('Gx', (0,), time=0.1)
        l2 = l.replace_name('Gz', 'Ga')
        self.assertEqual(l, l2)

    def test_labeltupwithtime_expand_subcircuits(self):
        l = LabelTupWithTime.init('Gx', (0,), time=0.1)
        self.assertEqual(l.expand_subcircuits(), (l,))

    def test_labeltupwithtime_add(self):
        l1 = LabelTupWithTime.init('Gx', (0,), time=0.1)
        l2 = l1 + '_foo'
        self.assertEqual(l2, LabelTupWithTime.init('Gx_foo', (0,), time=0.1))

        with self.assertRaises(NotImplementedError):
            l1 + l1


class LabelTupWithArgsTester(BaseCase):

    def test_labeltupwithargs(self):
        l = LabelTupWithArgs.init('Gx', (0,), args=('foo',))
        self.assertEqual(l.args, ('foo',))
        self.assertEqual(l.to_native(), ('Gx', 0, ';', 'foo'))
        self.assertEqual(l.name, 'Gx')
        self.assertEqual(l.sslbls, (0,))
        self.assertEqual(l.qubits, (0,))
        self.assertEqual(l.num_qubits, 1)
        self.assertEqual(l.components, (l,))

    def test_labeltupwithargs_replace_name(self):
        l = LabelTupWithArgs.init('Gx', (0,), args=('foo',))
        l2 = l.replace_name('Gx', 'Gy')
        self.assertEqual(l2, LabelTupWithArgs.init('Gy', (0,), args=('foo',)))

    def test_labeltupwithargs_lt_gt(self):
        l1 = LabelTupWithArgs.init('A', (0,), args=('foo',))
        l2 = LabelTupWithArgs.init('B', (0,), args=('foo',))
        l3 = LabelTupWithArgs.init('A', (1,), args=('foo',))
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)
        self.assertTrue(l1 < l3)

    def test_labeltupwithargs_map_state_space_labels(self):
        l = LabelTupWithArgs.init('Gx', (0, 1), args=('foo',))
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, LabelTupWithArgs.init('Gx', (1, 2), args=('foo',)))

    def test_labeltupwithargs_replace_name_no_match(self):
        l = LabelTupWithArgs.init('Gx', (0,), args=('foo',))
        l2 = l.replace_name('Gz', 'Ga')
        self.assertEqual(l, l2)

    def test_labeltupwithargs_expand_subcircuits(self):
        l = LabelTupWithArgs.init('Gx', (0,), args=('foo',))
        self.assertEqual(l.expand_subcircuits(), (l,))


class CircuitLabelTester(BaseCase):

    def test_circuitlabel_properties(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], (0,), 5, 1.2)
        self.assertEqual(l.name, 'mycirc')
        self.assertEqual(l.sslbls, (0,))
        self.assertEqual(l.qubits, (0,))
        self.assertEqual(l.num_qubits, 1)
        self.assertEqual(l.components, (L('Gx', 0),))
        self.assertEqual(l.reps, 5)
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args, ())
        self.assertEqual(l.depth, 5)

    def test_circuitlabel_expand_subcircuits(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], None, 2, 1.2)
        self.assertEqual(l.expand_subcircuits(), (L('Gx', 0), L('Gx', 0)))

        # a 2-layer sub-circuit, wrapped in a CircuitLabel, inside an args-bearing tuple-of-tuples
        sub = CircuitLabel('', [L('Gx', (0,)), L('Gy', (0,))], (0,), 1)
        top = LabelTupTupWithArgs.init((sub,), args=(0.5,))

        self.assertEqual(top.args, (0.5,))                                       # (0.5,)
        expanded = top.expand_subcircuits()
        self.assertIsInstance(expanded, tuple)
        self.assertFalse(isinstance(expanded, L))
        # This is no longer a label as labels are one layer unless they are CircuitLabels.
        for i in range(len(expanded)):
            self.assertEqual(expanded[i].args, (0.5,))

    def test_circuitlabel_expand_subcircuits_per_layer_args_with_reps(self):
        # A CircuitLabel whose two layers each carry their *own* args, with reps > 1.
        # Expanding must keep each layer's args with that layer (no cross-contamination)
        # and repeat the layer sequence `reps` times.
        layer0 = LabelTupTupWithArgs.init((('Gx', 0),), args=('a',))
        layer1 = LabelTupTupWithArgs.init((('Gy', 0),), args=('b',))
        cl = CircuitLabel('', [layer0, layer1], (0,), 2)

        self.assertEqual(cl.depth, 4)  # 2 layers * 2 reps
        expanded = cl.expand_subcircuits()
        self.assertEqual(len(expanded), 4)

        # Args follow their originating layer and repeat with reps; they do NOT
        # leak from one layer into the other.
        expected_args = [('a',), ('b',), ('a',), ('b',)]
        for lbl, exp_args in zip(expanded, expected_args):
            self.assertIsInstance(lbl, LabelTupTupWithArgs)
            self.assertEqual(lbl.args, exp_args)

        self.assertEqual(expanded[0].components[0].name, 'Gx')
        self.assertEqual(expanded[1].components[0].name, 'Gy')
        self.assertEqual(expanded[2].components[0].name, 'Gx')
        self.assertEqual(expanded[3].components[0].name, 'Gy')

    def test_circuitlabel_add(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], None, 5, 1.2)
        with self.assertRaises(NotImplementedError):
            l + "_foo"

    def test_circuitlabel_lt_gt(self):
        l1 = CircuitLabel('mycirc1', [L('Gx', 0)], None, 5, 1.2)
        l2 = CircuitLabel('mycirc2', [L('Gx', 0)], None, 5, 1.2)
        self.assertTrue(l1 < l2)
        self.assertFalse(l1 > l2)

    def test_circuitlabel_replace_name(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], (0,), 5, 1.2)
        l2 = l.replace_name('Gx', 'Gy')
        self.assertEqual(l2, CircuitLabel('mycirc', [L('Gy', 0)], (0,), 5, 1.2))

    def test_circuitlabel_map_state_space_labels(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], (0,), 5, 1.2)
        l2 = l.map_state_space_labels(lambda x: x + 1)
        self.assertEqual(l2, CircuitLabel('mycirc', [L('Gx', 1)], (1,), 5, 1.2))

    def test_circuitlabel_has_prefix(self):
        l = CircuitLabel('mycirc', [L('Gx', 0)], None, 5, 1.2)
        self.assertTrue(l.has_prefix('my'))
        self.assertFalse(l.has_prefix('your'))

    def test_circuitlabel_json_roundtrip_sslbls(self):
        # F10(c): json round-trip converts tuple state space labels into lists,
        # which needs to be parsed/coerced correctly on deserialization.
        import json
        cl = CircuitLabel('', (L('Gx', (0,)),), (0,), 1)
        native = cl.to_native()
        from_disk = json.loads(json.dumps(native))
        cl_rt = L(from_disk)
        self.assertEqual(cl_rt, cl)


class LabelConcateTester(BaseCase):

    def test_concat_labeltup(self):
        l1 = L(('Gx', 0))
        l2 = L(('Gy', 1))
        l3 = L((('Gz', 2), ('Ga', 3)))
        cl = CircuitLabel('circuit', [l3], None, 1, None)

        # LabelTup with LabelTup
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1))))

        # LabelTup with LabelTupTup
        concatenated = l1.concat(l3)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gz', 2), ('Ga', 3))))

        # LabelTup with CircuitLabel (depth 1)
        concatenated = l1.concat(cl)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gz', 2), ('Ga', 3))))

        # LabelTup with CircuitLabel (depth > 1)
        cl_depth2 = CircuitLabel('circuit', [l1, l2], None, 1, None)
        with self.assertRaises(ValueError):
            l1.concat(cl_depth2)

        # LabelTup with LabelStr
        l_str = L('Idle')
        with self.assertRaises(ValueError):
            l1.concat(l_str)

    def test_concat_args_do_not_duplicate(self):
        a = LabelTupWithArgs.init('Gx', (0,), args=(0.5,))
        b = LabelTupWithArgs.init('Gy', (1,), args=(0.7,))
        r = a.concat(b)

        self.assertNotEqual(r.args, r.collect_args())
        self.assertEqual(len(r.args), 0)

    def test_concat_labeltupwithtime(self):
        l1 = L(('Gx', 0), time=0.1)
        l2 = L(('Gy', 1))
        l3 = L((('Gz', 2), ('Ga', 3)))
        cl = CircuitLabel('circuit', [l3], None, 1, None)

        # LabelTupWithTime with LabelTup
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1))))

        # LabelTupWithTime with LabelTupTup
        concatenated = l1.concat(l3)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gz', 2), ('Ga', 3))))

        # LabelTupWithTime with CircuitLabel (depth 1)
        concatenated = l1.concat(cl)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gz', 2), ('Ga', 3))))

        # LabelTupWithTime with CircuitLabel (depth > 1)
        cl_depth2 = CircuitLabel('circuit', [l1, l2], None, 1, None)
        with self.assertRaises(ValueError):
            l1.concat(cl_depth2)

        # LabelTupWithTime with LabelStr
        l_str = L('Idle')
        with self.assertRaises(ValueError):
            l1.concat(l_str)

        l1 = L(('Gx', 0), time=0.1)
        l2 = L(('Gy', 1), time=0.2)
        from pygsti.tools.exceptions import ClobberingWarning
        with self.assertWarns(ClobberingWarning):
            l1.concat(l2)

        l1 = L(('Gx', 0), time=0.1)
        l2 = L(('Gy', 1), time=0.1)
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1))))
        return

    def test_concat_circuitlabel_depth(self):
        l1 = L(('Gx', 0))
        cl_depth2 = CircuitLabel('circuit', ["Gx", "Gy"], (2,), 1, None)
        with self.assertRaises(ValueError):
            cl_depth2.concat(l1)

    def test_concat_circuit_label(self):
        lbl = CircuitLabel("foo", [(L("Gx",0), L("Gy",1))], state_space_labels=(0,1))
        lbl2 = L("Gz", 2)
        concatenated = lbl.concat(lbl2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((("Gx",0),("Gy",1), ("Gz",2))))

        concatenated = lbl2.concat(lbl)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((("Gz",2), ("Gx",0),("Gy",1))))

    def test_concat_labeltuptup(self):
        l1 = L((('Gx', 0), ('Gy', 1)))
        l2 = L(('Gz', 2))
        l3 = L((('Ga', 3), ('Gb', 4)))
        cl = CircuitLabel('circuit', [l3], None, 1, None)

        # LabelTupTup with CircuitLabel depth 1
        concatenated = l1.concat(cl)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1), ('Ga', 3), ('Gb', 4))))

        # LabelTupTup with CircuitLabel depth > 1
        cl_depth2 = CircuitLabel('circuit', [l2, l3], None, 1, None)
        with self.assertRaises(ValueError):
            l1.concat(cl_depth2)

        # LabelTupTup with LabelTup
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1), ('Gz', 2))))

        # LabelTupTup with LabelTupTup
        concatenated = l1.concat(l3)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1), ('Ga', 3), ('Gb', 4))))

        # LabelTupTup with LabelStr
        l_str = L('Idle')
        with self.assertRaises(ValueError):
            l1.concat(l_str)

    def test_concat_labeltuptupwithtime(self):
        l1 = L((('Gx', 0), ('Gy', 1)), time=0.1)
        l2 = L((('Gz', 2),), time=0.1)
        l3 = L((('Ga', 3),), time=0.2)

        # LabelTupTupWithTime with LabelTupTupWithTime (same time)
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1), ('Gz', 2))))

        # LabelTupTupWithTime with LabelTupTupWithTime (different time)
        from pygsti.tools.exceptions import ClobberingWarning
        with self.assertWarns(ClobberingWarning):
            l1.concat(l3)
        # LabelTupTupWithTime with LabelTupTupWithTime (different time)
        with self.assertWarns(ClobberingWarning):
            l3.concat(l1)

        # LabelTupTupWithTime with LabelTupWithTime (same time)
        l_tup_with_time = L(('Gb', 4), time=0.1)
        concatenated = l1.concat(l_tup_with_time)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((('Gx', 0), ('Gy', 1), ('Gb', 4))))

        # LabelTupTupWithTime with LabelTupWithTime (different time)
        l_tup_with_time_different = L(('Gc', 5), time=0.2)
        with self.assertWarns(ClobberingWarning):
            l1.concat(l_tup_with_time_different)

    def test_concat_qubit_overlap(self):
        l1 = L(('Gx', 0))
        l2 = L(('Gy', 0))
        with self.assertRaises(ValueError):
            l1.concat(l2)

        l3 = L((('Gx', 0), ('Gy', 1)))
        l4 = L(('Gz', 1))
        with self.assertRaises(ValueError):
            l3.concat(l4)

    def test_concat_always_returns_labeltuptup(self):
        # A successful concat always produces a (plain) LabelTupTup, even when
        # the operands carry a time value.  Time metadata is not propagated.
        l1 = L(('Gx', 0), time=0.5)
        l2 = L(('Gy', 1), time=0.5)
        self.assertTrue(isinstance(l1, LabelTupWithTime))
        concatenated = l1.concat(l2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertNotIsInstance(concatenated, LabelTupTupWithTime)
        self.assertFalse(hasattr(concatenated, 'time'))
        self.assertEqual(concatenated.collect_args(), l1.collect_args() + l2.collect_args())

        l3 = L([('Gz', 2)], time=0.5)
        concatenated2 = concatenated.concat(l3)
        self.assertIsInstance(concatenated2, LabelTupTup)
        self.assertNotIsInstance(concatenated2, LabelTupTupWithTime)
        self.assertFalse(hasattr(concatenated2, 'time'))
        self.assertEqual(concatenated2.collect_args(), l1.collect_args() + l2.collect_args() + l3.collect_args())

    def test_concat_does_not_warn_on_consistent_component_time(self):
        # concat deliberately produces a timeless LabelTupTup, so it must not
        # emit the "consider a LabelTupTupWithTime" hint that direct
        # construction does when all components share a non-zero time.
        import warnings
        l1 = L(('Gx', 0), time=0.1)
        l2 = L(('Gy', 1), time=0.1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            l1.concat(l2)
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    self.assertNotIn("You may want to consider a LabelTupTupWithTime",
                                     str(warning.message))

    def test_concat_labelstr_errors(self):

        ltt_wt = L((('Gx', 0), ('Gy', 1)), time=0.1)
        ltt = L((('Gx', 0), ('Gy', 1)))
        lt_wt = L(('Gy', 1), time=0.1)
        lt = L(('Gy', 1))
        cl = CircuitLabel('circuit', [lt], (1,), 1, None)

        mystr = L("Gx")
        self.assertIsInstance(mystr, LabelStr)

        with self.assertRaises(ValueError):
            mystr.concat(ltt_wt)
        with self.assertRaises(ValueError):
            mystr.concat(ltt)
        with self.assertRaises(ValueError):
            mystr.concat(lt_wt)
        with self.assertRaises(ValueError):
            mystr.concat(lt)
        with self.assertRaises(ValueError):
            mystr.concat(cl)

    def test_concat_two_circuitlabels(self):
        c1 = CircuitLabel('a', [L('Gx', 0)], (0,), 1, None)
        c2 = CircuitLabel('b', [L('Gy', 1)], (1,), 1, None)

        # Equal depth: concatenation yields a LabelTupTup of the two CircuitLabels.
        concatenated = c1.concat(c2)
        self.assertIsInstance(concatenated, LabelTupTup)
        self.assertEqual(concatenated, L((c1, c2)))

        # Different depth: raises ValueError.
        c_depth2 = CircuitLabel('d', [L('Gx', 2)], (2,), 2, None)
        with self.assertRaises(ValueError):
            c1.concat(c_depth2)

    def test_concat_order_matters(self):
        l1 = L(('Gx', 0))
        l2 = L(('Gy', 1))
        concatenated1 = l1.concat(l2)
        concatenated2 = l2.concat(l1)
        self.assertNotEqual(concatenated1, concatenated2)

    def test_concat_circuitlabel_preserved_iff_both_circuitlabels(self):
        # For a depth-1 CircuitLabel, lbl1.concat(lbl2) retains a CircuitLabel in
        # the result *if and only if* BOTH operands are depth-1 CircuitLabels.
        # If either operand is not a CircuitLabel, the depth-1 CircuitLabel is
        # unwrapped and the result is a flat LabelTupTup with no CircuitLabel.
        def has_circuitlabel(lbl):
            return any(isinstance(c, CircuitLabel) for c in lbl.components)

        cl1 = CircuitLabel('a', [L('Gx', 0)], (0,), 1, None)
        cl2 = CircuitLabel('b', [L('Gy', 1)], (1,), 1, None)
        lt = L(('Gz', 2))
        ltt = L((('Ga', 3), ('Gb', 4)))

        # Both operands depth-1 CircuitLabels -> CircuitLabels are preserved.
        both = cl1.concat(cl2)
        self.assertIsInstance(both, LabelTupTup)
        self.assertTrue(has_circuitlabel(both))
        self.assertEqual(both.components, (cl1, cl2))

        # Exactly one operand is a (depth-1) CircuitLabel -> no CircuitLabel remains,
        # regardless of which operand it is or what non-CircuitLabel it's combined with.
        non_circuitlabels = [lt, ltt]
        for other in non_circuitlabels:
            for result in (cl1.concat(other), other.concat(cl1)):
                self.assertIsInstance(result, LabelTupTup)
                self.assertFalse(has_circuitlabel(result))
