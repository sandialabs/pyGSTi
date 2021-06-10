from pygsti.construction.circuitconstruction import to_circuits
from pygsti.modelpacks import smq1Q_XYZI, smq2Q_XYICPHASE
from ..util import BaseCase


class ModelpackBase:
    def test_indexed_circuits(self):
        fiducials_default = self.modelpack.fiducials()
        self.assertEqual(fiducials_default, self.expected_fiducials_default)

        fiducials_new_idx = self.modelpack.fiducials([n + 10 for n in self.modelpack._sslbls])
        self.assertEqual(fiducials_new_idx, self.expected_fiducials_new_idx)


class Modelpack1QTester(ModelpackBase, BaseCase):
    modelpack = smq1Q_XYZI
    expected_fiducials_default = to_circuits(
        [(), (('Gxpi2', 0), ), (('Gypi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
         (('Gypi2', 0), ('Gypi2', 0), ('Gypi2', 0))], line_labels=[0]
    )
    expected_fiducials_new_idx = to_circuits(
        [(), (('Gxpi2', 10), ), (('Gypi2', 10), ), (('Gxpi2', 10), ('Gxpi2', 10)), (('Gxpi2', 10), ('Gxpi2', 10), ('Gxpi2', 10)),
         (('Gypi2', 10), ('Gypi2', 10), ('Gypi2', 10))], line_labels=[10]
    )


class Modelpack2QTester(ModelpackBase, BaseCase):
    modelpack = smq2Q_XYICPHASE
    expected_fiducials_default = to_circuits(
        [(), (('Gxpi2', 1), ), (('Gypi2', 1), ), (('Gxpi2', 1), ('Gxpi2', 1)), (('Gxpi2', 0), ), (('Gxpi2', 0), ('Gxpi2', 1)),
         (('Gxpi2', 0), ('Gypi2', 1)), (('Gxpi2', 0), ('Gxpi2', 1), ('Gxpi2', 1)), (('Gypi2', 0), ), (('Gypi2', 0), ('Gxpi2', 1)),
         (('Gypi2', 0), ('Gypi2', 1)), (('Gypi2', 0), ('Gxpi2', 1), ('Gxpi2', 1)), (('Gxpi2', 0), ('Gxpi2', 0)),
         (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 1)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gypi2', 1)),
         (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 1), ('Gxpi2', 1))], line_labels=[0, 1]
    )
    expected_fiducials_new_idx = to_circuits(
        [(), (('Gxpi2', 11), ), (('Gypi2', 11), ), (('Gxpi2', 11), ('Gxpi2', 11)), (('Gxpi2', 10), ), (('Gxpi2', 10), ('Gxpi2', 11)),
         (('Gxpi2', 10), ('Gypi2', 11)), (('Gxpi2', 10), ('Gxpi2', 11), ('Gxpi2', 11)), (('Gypi2', 10), ), (('Gypi2', 10), ('Gxpi2', 11)),
         (('Gypi2', 10), ('Gypi2', 11)), (('Gypi2', 10), ('Gxpi2', 11), ('Gxpi2', 11)), (('Gxpi2', 10), ('Gxpi2', 10)),
         (('Gxpi2', 10), ('Gxpi2', 10), ('Gxpi2', 11)), (('Gxpi2', 10), ('Gxpi2', 10), ('Gypi2', 11)),
         (('Gxpi2', 10), ('Gxpi2', 10), ('Gxpi2', 11), ('Gxpi2', 11))], line_labels=[10, 11]
    )
