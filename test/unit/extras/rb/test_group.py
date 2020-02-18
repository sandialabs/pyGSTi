from ...util import BaseCase

#from pygsti.extras.rb import group

class RBGroupTester(BaseCase):
    def test_construct_1Q_Clifford_group(self):
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        clifford = group.construct_1Q_Clifford_group()
        # TODO assert correctness
