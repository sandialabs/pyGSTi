import pickle

import pygsti.baseobjs.outcomelabeldict as ld
from pygsti.baseobjs.label import Label
from pygsti.models.memberdict import OrderedMemberDict
from pygsti.models.modelconstruction import create_explicit_model_from_expressions
from pygsti.models import ExplicitOpModel
from ..util import BaseCase


class LabelDictTester(BaseCase):
    def test_ordered_member_dict(self):
        flags = {'auto_embed': True, 'match_parent_dim': True,
                 'match_parent_evotype': True, 'cast_to_type': "spamvec"}
        d = OrderedMemberDict(None, "foobar", "rho", flags)
        #print(d.items())
        #assert False
        # TODO assert correctness

        with self.assertRaises(ValueError):
            d['rho0'] = [0]  # bad default parameter type

    def test_iter_gatesets(self):
        model = create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        model2 = ExplicitOpModel(['Q0'])
        for label, gate in model.operations.items():
            model2[label] = gate.copy()
        for label, vec in model.preps.items():
            model2[label] = vec.copy()
        for label, povm in model.povms.items():
            model2[label] = povm.copy()

        self.assertAlmostEqual(model.frobeniusdist(model2), 0.0)

    def test_outcome_label_dict(self):
        d = ld.OutcomeLabelDict([(('0',), 90), (('1',), 10)])
        self.assertEqual(d['0'], 90)  # don't need tuple when they're 1-tuples
        self.assertEqual(d['1'], 10)  # don't need tuple when they're 1-tuples

    def test_outcome_label_dict_pickles(self):
        d = ld.OutcomeLabelDict([(('0',), 90), (('1',), 10)])
        s = pickle.dumps(d)
        d_pickle = pickle.loads(s)
        self.assertEqual(d, d_pickle)

    def test_outcome_label_dict_copy(self):
        d = ld.OutcomeLabelDict([(('0',), 90), (('1',), 10)])
        d_copy = d.copy()
        self.assertEqual(d, d_copy)

    def test_validate_keys_round_trip(self):
        # Build a real ExplicitOpModel so we have valid ModelMember values to assign.
        # The four primary member dicts (preps, povms, operations, instruments)
        # have validate_keys=True; factories does not.
        model = create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        # Grab a real Instrument-compatible member by constructing a tiny
        # ExplicitOpModel-style instrument from existing ops via the
        # public Instrument class.
        from pygsti.modelmembers import instruments as _inst
        op_gi = model.operations[Label('Gi')]
        instr_value = _inst.Instrument([('outcome', op_gi.copy())])

        # 1) Round-tripping str key -> success
        model.instruments[Label('Iz')] = instr_value
        # 2) Non-round-tripping str key -> ValueError
        with self.assertRaises(ValueError):
            model.instruments['IzTP'] = instr_value
        # 3) Label constructed from non-round-tripping string -> ValueError
        with self.assertRaises(ValueError):
            model.instruments[Label('IzTP')] = instr_value
        # 4) Label constructed from round-tripping string -> success
        model.instruments[Label('Iz2')] = instr_value

        # 5) Opt-in nature: when validate_keys is False (default),
        # the same assignments succeed on a bare OrderedMemberDict.
        d = OrderedMemberDict(None, "full", "I", {'cast_to_type': None})
        # Use a ModelMember directly to bypass cast logic.
        d[Label('IzTP')] = op_gi.copy()
        d['I_zTP'] = op_gi.copy()
        return

    def test_multi_prefix_ordered_member_dict(self):
        model = create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        op_gi = model.operations[Label('Gi')]

        # Multi-prefix dictionary accepting 'G' and '{'
        d = OrderedMemberDict(None, "full", ('G', '{'), {'cast_to_type': None})
        d[Label('Gx')] = op_gi.copy()
        d['Gy'] = op_gi.copy()
        d[Label('{auto_global_idle}')] = op_gi.copy()
        d['{idle}'] = op_gi.copy()

        self.assertIn(Label('Gx'), d)
        self.assertIn(Label('{auto_global_idle}'), d)
        self.assertIn('Gy', d)
        self.assertIn('{idle}', d)

        # Invalid insertions should raise KeyError with informative message
        with self.assertRaises(KeyError) as cm:
            d['rho0'] = op_gi.copy()
        self.assertIn("one of the prefixes 'G', '{'", str(cm.exception))

        with self.assertRaises(KeyError) as cm:
            d[Label('Mdefault')] = op_gi.copy()
        self.assertIn("one of the prefixes 'G', '{'", str(cm.exception))

        with self.assertRaises(KeyError) as cm:
            d['I0'] = op_gi.copy()
        self.assertIn("one of the prefixes 'G', '{'", str(cm.exception))

        # Copy preserves multi-prefix policy
        d_copy = d.copy()
        self.assertEqual(d_copy._prefix, ('G', '{'))
        d_copy[Label('Gz')] = op_gi.copy()
        d_copy['{new_idle}'] = op_gi.copy()
        with self.assertRaises(KeyError):
            d_copy['rho0'] = op_gi.copy()

        # Pickle preserves multi-prefix policy
        d_pickled = pickle.loads(pickle.dumps(d))
        self.assertEqual(d_pickled._prefix, ('G', '{'))
        self.assertIn(Label('Gx'), d_pickled)
        self.assertIn(Label('{auto_global_idle}'), d_pickled)
        d_pickled[Label('Gz')] = op_gi.copy()
        d_pickled['{new_idle}'] = op_gi.copy()
        with self.assertRaises(KeyError):
            d_pickled['rho0'] = op_gi.copy()

        # Item-based initialization
        items = [(Label('Gx'), op_gi.copy()), (Label('{idle}'), op_gi.copy())]
        d_init = OrderedMemberDict(None, "full", ('G', '{'), {'cast_to_type': None}, items=items)
        self.assertIn(Label('Gx'), d_init)
        self.assertIn(Label('{idle}'), d_init)

        bad_items = [(Label('rho0'), op_gi.copy())]
        with self.assertRaises(KeyError):
            OrderedMemberDict(None, "full", ('G', '{'), {'cast_to_type': None}, items=bad_items)

        # Unconstrained prefix=None accepts anything
        d_none = OrderedMemberDict(None, "full", None, {'cast_to_type': None})
        d_none['rho0'] = op_gi.copy()
        d_none['Mdefault'] = op_gi.copy()
        d_none['Gx'] = op_gi.copy()
        d_none['{idle}'] = op_gi.copy()
        self.assertIn('rho0', d_none)
        self.assertIn('Mdefault', d_none)
        self.assertIn('Gx', d_none)
        self.assertIn('{idle}', d_none)

        # Single prefix retains single prefix behavior and error message
        d_single = OrderedMemberDict(None, "full", "rho", {'cast_to_type': None})
        d_single['rho0'] = op_gi.copy()
        with self.assertRaises(KeyError) as cm:
            d_single['Gx'] = op_gi.copy()
        self.assertIn("beginning with the prefix 'rho'", str(cm.exception))
