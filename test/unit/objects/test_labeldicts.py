from ..util import BaseCase

from pygsti.construction.modelconstruction import build_explicit_model
from pygsti.objects import ExplicitOpModel
import pygsti.objects.labeldicts as ld


class LabelDictTester(BaseCase):
    def test_ordered_member_dict(self):
        flags = {'auto_embed': True, 'match_parent_dim': True,
                 'match_parent_evotype': True, 'cast_to_type': "spamvec"}
        d = ld.OrderedMemberDict(None, "foobar", "rho", flags)
        # TODO assert correctness

        with self.assertRaises(ValueError):
            d['rho0'] = [0]  # bad default parameter type

    def test_iter_gatesets(self):
        model = build_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        model2 = ExplicitOpModel(['Q0'])
        for label, gate in model.operations.items():
            model2[label] = gate
        for label, vec in model.preps.items():
            model2[label] = vec
        for label, povm in model.povms.items():
            model2[label] = povm

        self.assertAlmostEqual(model.frobeniusdist(model2), 0.0)
