import numpy as np

from ..util import BaseCase

from pygsti.objects import modelmember as mm


class ModelMemberUtilTester(BaseCase):
    def test_compose_gpindices(self):
        parent_gpindices = slice(10, 20)
        child_gpindices = slice(2, 4)
        x = mm._compose_gpindices(parent_gpindices, child_gpindices)
        self.assertEqual(x, slice(12, 14))

        parent_gpindices = slice(10, 20)
        child_gpindices = np.array([0, 2, 4], 'i')
        x = mm._compose_gpindices(parent_gpindices, child_gpindices)
        self.assertEqual(list(x), list(np.array([10, 12, 14], 'i')))  # lists so assertEqual works

        parent_gpindices = np.array([2, 4, 6, 8, 10], 'i')
        child_gpindices = np.array([0, 2, 4], 'i')
        x = mm._compose_gpindices(parent_gpindices, child_gpindices)
        self.assertEqual(list(x), list(np.array([2, 6, 10], 'i')))

    def test_decompose_gpindices(self):
        parent_gpindices = slice(10, 20)
        sibling_gpindices = slice(12, 14)
        x = mm._decompose_gpindices(parent_gpindices, sibling_gpindices)
        self.assertEqual(x, slice(2, 4))

        parent_gpindices = slice(10, 20)
        sibling_gpindices = np.array([10, 12, 14], 'i')
        x = mm._decompose_gpindices(parent_gpindices, sibling_gpindices)
        self.assertEqual(list(x), list(np.array([0, 2, 4], 'i')))

        parent_gpindices = np.array([2, 4, 6, 8, 10], 'i')
        sibling_gpindices = np.array([2, 6, 10], 'i')
        x = mm._decompose_gpindices(parent_gpindices, sibling_gpindices)
        self.assertEqual(list(x), list(np.array([0, 2, 4], 'i')))


class ModelMemberInstanceTester(BaseCase):
    def setUp(self):
        self.member = mm.ModelMember(dim=4, evotype="densitymx")

    def test_gpindices_read_only(self):
        with self.assertRaises(ValueError):
            self.member.gpindices = slice(0, 3)  # read-only!
        with self.assertRaises(ValueError):
            self.member.parent = None  # read-only!
