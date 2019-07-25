from ...util import BaseCase

from pygsti.extras.rb import group


class RBGroupTester(BaseCase):
    def test_construct_1Q_Clifford_group(self):
        clifford = group.construct_1Q_Clifford_group()
        # TODO assert correctness
