from pygsti.models import qutrit
from ..util import BaseCase


class QutritConstructionTester(BaseCase):
    def test_ideal_qutrit(self):
        mdl_post = qutrit.create_qutrit_model(error_scale=0.0, similarity=False, seed=1234, basis='qt')
        mdl_sim = qutrit.create_qutrit_model(error_scale=0.0, similarity=True, seed=1234, basis='qt')
        self.assertAlmostEqual(mdl_sim.frobeniusdist(mdl_post), 0)
    
    def test_noisy_qutrit(self):
        mdl_sim = qutrit.create_qutrit_model(error_scale=0.1, similarity=True, seed=1234, basis='qt')
        mdl_ideal = qutrit.create_qutrit_model(error_scale=0.1, similarity=True, seed=1234, basis='qt')
        self.assertArraysAlmostEqual(mdl_sim['Gi', 'QT'].to_dense(), mdl_ideal['Gi', 'QT'].to_dense())

        #just test building a gate in the qutrit basis
        # Can't do this b/c need a 'T*' triplet space designator for "triplet space" and it doesn't seem
        # worth adding this now...
        #qutrit_gate = pygsti.construction._create_operation( [1,1,1], [('L0',),('L1',),('L2',)], "I()", 'qt', 'full')
