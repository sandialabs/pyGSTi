from ..util import BaseCase

import pygsti
from pygsti.construction import qutrit


class QutritConstructionTester(BaseCase):
    def test_qutrit_gateset(self):
        mdl = qutrit.create_qutrit_model(error_scale=0.1, similarity=False, seed=1234, basis='qt')
        gs2 = qutrit.create_qutrit_model(error_scale=0.1, similarity=True, seed=1234, basis='qt')
        # TODO assert correctness

        #just test building a gate in the qutrit basis
        # Can't do this b/c need a 'T*' triplet space designator for "triplet space" and it doesn't seem
        # worth adding this now...
        #qutrit_gate = pygsti.construction._create_operation( [1,1,1], [('L0',),('L1',),('L2',)], "I()", 'qt', 'full')
