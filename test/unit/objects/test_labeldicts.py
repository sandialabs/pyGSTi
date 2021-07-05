import pickle

import pygsti.baseobjs.outcomelabeldict as ld
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
            model2[label] = gate
        for label, vec in model.preps.items():
            model2[label] = vec
        for label, povm in model.povms.items():
            model2[label] = povm

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
