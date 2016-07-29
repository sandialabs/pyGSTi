from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.objects import GateString

import pygsti.construction.rpeconstruction as rc
import pygsti.construction as pc

import unittest

from pprint import pprint

class RPEConstructionTestCase(BaseTestCase):

    def setUp(self):
        super(RPEConstructionTestCase, self).setUp()
        self.lengths = [2, 4, 8, 16, 32]

    def build_lists(self, fids1, fids2, germ):
        lists = ([GateString(fids1 % (germ + str(length))) for length in self.lengths],
                 [GateString(fids2 % (germ + str(length))) for length in self.lengths])
        return lists

    def to_tuples(self, l1, l2):
        # Convert gatestrings to tuples for comparison
        def toTuples(ls):
            tuples = []
            for item in ls:
                tuples.append(tuple(item))
        l1 = (toTuples(l1[0]), toTuples(l1[1]))
        l2 = (toTuples(l2[0]), toTuples(l2[1]))
        return l1, l2

    # I'm assuming these angles are in radians, based on documentation.
    def test_make_parameterized_rpe_gateset(self):
        # These numbers have no significance
        A = rc.make_parameterized_rpe_gate_set(1.57079632679, 1.57079632679, .78539816339, 0.001, 0.001)
        B = rc.make_parameterized_rpe_gate_set(1.57079632679, 1.57079632679, .78539816339, 0.001, 0.001)
        self.assertEqual(A.frobeniusdist(B), 0.0)

        # Again, no significance in these numbers
        C = rc.make_parameterized_rpe_gate_set(1.56079632679, 1.56079632679, .78539816339, 0.001, 0.001, True)
        self.assertAlmostEqual(A.frobeniusdist(C), 0.0, 2)

    def test_make_rpe_alpha_str_lists_gx_gz(self): # At least we can be sure about what this function is doing
        lists           = rc.make_rpe_alpha_str_lists_gx_gz(self.lengths)
        expected        = self.build_lists('GiGxGxGz%sGzGzGxGx', 'GxGxGzGz%sGzGzGzGxGx', 'Gz^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_rpe_epsilon_str_lists_gx_gz(self):
        lists           = rc.make_rpe_epsilon_str_lists_gx_gz(self.lengths)
        expected        = self.build_lists('%sGxGxGxGx', 'GxGxGzGz%sGxGxGxGx', 'Gx^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_make_rpe_theta_str_lists_gx_gz(self):
        lists           = rc.make_rpe_theta_str_lists_gx_gz(self.lengths)
        expected        = self.build_lists('%sGxGxGxGx', '(GxGxGzGz)%sGxGxGxGx', '(GzGxGxGxGxGzGzGxGxGxGxGz)^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_make_rpe_string_list_d(self):
        d = rc.make_rpe_string_list_d(3)

    def test_make_rpe_data_set(self):
        A = rc.make_parameterized_rpe_gate_set(1.57079632679, 1.57079632679, .78539816339, 0.001, 0.001)
        d = rc.make_rpe_string_list_d(3)
        rc.make_rpe_data_set(A, d, 1000)

    def test_ensemble(self):
        # Just make sure no errors get thrown...
        rc.rpe_ensemble_test(1.57079632679, 1.57079632679, .78539816339, 0.001, 3, 1, 1)
        rc.rpe_ensemble_test(1.57079632679, 1.57079632679, .78539816339, 0.001, 3, 1, 1, plot=True)





if __name__ == '__main__':
    unittest.main(verbosity=2)
