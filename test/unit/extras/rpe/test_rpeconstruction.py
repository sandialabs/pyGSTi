import pygsti.extras.rpe.rpeconstruction as rc
from pygsti.circuits import Circuit
from ...util import BaseCase


class RPEConstructionFuncBase(object):
    lengths = [2, 4, 8, 16, 32]

    def build_lists(self, fids1, fids2, germ):
        lists = ([Circuit(fids1 % (germ + str(length))) for length in self.lengths],
                 [Circuit(fids2 % (germ + str(length))) for length in self.lengths])
        return lists

    def to_tuples(self, l1, l2):
        # Convert circuits to tuples for comparison
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
        A = rc.create_parameterized_rpe_model(
            1.57079632679, 1.57079632679, 0.78539816339, 0.001, 0.001,
            rpeconfig_inst=self.config
        )
        B = rc.create_parameterized_rpe_model(
            1.57079632679, 1.57079632679, 0.78539816339, 0.001, 0.001,
            rpeconfig_inst=self.config
        )
        self.assertEqual(A.frobeniusdist(B), 0.0)

        # Again, no significance in these numbers
        C = rc.create_parameterized_rpe_model(
            1.56079632679, 1.56079632679, 0.78539816339, 0.001, 0.001,
            True, rpeconfig_inst=self.config
        )
        self.assertAlmostEqual(A.frobeniusdist(C), 0.0, 2)

    # At least we can be sure about what this function is doing
    def test_make_rpe_alpha_str_lists_gx_gz(self):
        lists = rc.create_rpe_angle_circuit_lists(self.lengths, "alpha", self.config)
        expected = self.build_lists('GiGxGxGz%sGzGzGxGx', 'GxGxGzGz%sGzGzGzGxGx', 'Gz^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_rpe_epsilon_str_lists_gx_gz(self):
        lists = rc.create_rpe_angle_circuit_lists(self.lengths, "epsilon", self.config)
        expected = self.build_lists('%sGxGxGxGx', 'GxGxGzGz%sGxGxGxGx', 'Gx^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_make_rpe_theta_str_lists_gx_gz(self):
        lists = rc.create_rpe_angle_circuit_lists(self.lengths, "theta", self.config)
        expected = self.build_lists('%sGxGxGxGx', '(GxGxGzGz)%sGxGxGxGx', '(GzGxGxGxGxGzGzGxGxGxGxGz)^')
        lists, expected = self.to_tuples(lists, expected)
        self.assertEqual(lists, expected)

    def test_make_rpe_string_list_dict(self):
        stringListD = rc.create_rpe_angle_circuits_dict(2, self.config)
        # TODO assert correctness

    def test_make_rpe_data_set(self):
        A = rc.create_parameterized_rpe_model(
            1.57079632679, 1.57079632679, 0.78539816339, 0.001, 0.001,
            rpeconfig_inst=self.config
        )
        d = rc.create_rpe_angle_circuits_dict(3, self.config)
        rpreDS = rc.create_rpe_dataset(A, d, 1000)
        # TODO assert correctness


class RPEConstruction00ConfigTester(RPEConstructionFuncBase, BaseCase):
    from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_00 as config  # noqa N813


class RPEConstructionUpDnConfigTester(RPEConstructionFuncBase, BaseCase):
    from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_UpDn as config  # noqa N813
