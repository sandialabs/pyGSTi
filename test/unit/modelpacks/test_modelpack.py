from ..util import BaseCase

from pygsti.construction.circuitconstruction import circuit_list
from pygsti.modelpacks import smq1Q_XYZI, smq2Q_XYICPHASE


class ModelpackBase:
    def test_indexed_circuits(self):
        fiducials_default = self.modelpack.fiducials()
        self.assertEqual(fiducials_default, self.expected_fiducials_default)

        fiducials_new_idx = self.modelpack.fiducials(lambda n: n + 10)
        self.assertEqual(fiducials_new_idx, self.expected_fiducials_new_idx)


class Modelpack1QTester(ModelpackBase, BaseCase):
    modelpack = smq1Q_XYZI
    expected_fiducials_default = circuit_list(
        [(), (('Gx', 0), ), (('Gy', 0), ), (('Gx', 0), ('Gx', 0)), (('Gx', 0), ('Gx', 0), ('Gx', 0)),
         (('Gy', 0), ('Gy', 0), ('Gy', 0))], line_labels=[0]
    )
    expected_fiducials_new_idx = circuit_list(
        [(), (('Gx', 10), ), (('Gy', 10), ), (('Gx', 10), ('Gx', 10)), (('Gx', 10), ('Gx', 10), ('Gx', 10)),
         (('Gy', 10), ('Gy', 10), ('Gy', 10))], line_labels=[10]
    )


class Modelpack2QTester(ModelpackBase, BaseCase):
    modelpack = smq2Q_XYICPHASE
    expected_fiducials_default = circuit_list(
        [(), (('Gx', 1), ), (('Gy', 1), ), (('Gx', 1), ('Gx', 1)), (('Gx', 0), ), (('Gx', 0), ('Gx', 1)),
         (('Gx', 0), ('Gy', 1)), (('Gx', 0), ('Gx', 1), ('Gx', 1)), (('Gy', 0), ), (('Gy', 0), ('Gx', 1)),
         (('Gy', 0), ('Gy', 1)), (('Gy', 0), ('Gx', 1), ('Gx', 1)), (('Gx', 0), ('Gx', 0)),
         (('Gx', 0), ('Gx', 0), ('Gx', 1)), (('Gx', 0), ('Gx', 0), ('Gy', 1)),
         (('Gx', 0), ('Gx', 0), ('Gx', 1), ('Gx', 1))], line_labels=[0, 1]
    )
    expected_fiducials_new_idx = circuit_list(
        [(), (('Gx', 11), ), (('Gy', 11), ), (('Gx', 11), ('Gx', 11)), (('Gx', 10), ), (('Gx', 10), ('Gx', 11)),
         (('Gx', 10), ('Gy', 11)), (('Gx', 10), ('Gx', 11), ('Gx', 11)), (('Gy', 10), ), (('Gy', 10), ('Gx', 11)),
         (('Gy', 10), ('Gy', 11)), (('Gy', 10), ('Gx', 11), ('Gx', 11)), (('Gx', 10), ('Gx', 10)),
         (('Gx', 10), ('Gx', 10), ('Gx', 11)), (('Gx', 10), ('Gx', 10), ('Gy', 11)),
         (('Gx', 10), ('Gx', 10), ('Gx', 11), ('Gx', 11))], line_labels=[10, 11]
    )
