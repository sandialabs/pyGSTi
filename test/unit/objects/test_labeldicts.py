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
