import unittest

import numpy as np

import pygsti
import pygsti.extras.rpe.rpeconstruction as rc
from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_00
from pygsti.extras.rpe import rpeconfig_GxPi2_GyPi2_UpDn
from ..testutils import BaseTestCase


class TestRPEObjectMethods(BaseTestCase):

    def test_rpe_datasets(self):

        model = pygsti.construction.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy', 'Gz'],
                                                          [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)", "Z(pi/2,Q0)"])

        depol_gateset = model.depolarize(op_noise=0.1,spam_noise=0)

        #test RPE datasets
        rpeconfig_inst_list = [rpeconfig_GxPi2_GyPi2_UpDn,rpeconfig_GxPi2_GyPi2_00]
        for rpeconfig_inst in rpeconfig_inst_list:
            rpeGS  = rc.create_parameterized_rpe_model(np.pi/2, np.pi/4, 0, 0.1, 0.1, True, rpeconfig_inst=rpeconfig_inst)
            rpeGS2 = rc.create_parameterized_rpe_model(np.pi/2, np.pi/4, 0, 0.1, 0.1, False, rpeconfig_inst=rpeconfig_inst)
            rpeGS3 = rc.create_parameterized_rpe_model(np.pi/2, np.pi/4, np.pi/4, 0.1, 0.1, False, rpeconfig_inst=rpeconfig_inst)

            kList = [0,1,2]
            lst1 = rc.create_rpe_angle_circuit_lists(kList, "alpha", rpeconfig_inst)
            lst2 = rc.create_rpe_angle_circuit_lists(kList, "epsilon", rpeconfig_inst)
            lst3 = rc.create_rpe_angle_circuit_lists(kList, "theta", rpeconfig_inst)
            lstDict = rc.create_rpe_angle_circuits_dict(2,rpeconfig_inst)

            rpeDS = rc.create_rpe_dataset(depol_gateset,lstDict,1000,
                                         sample_error='binomial',seed=1234)


if __name__ == "__main__":
    unittest.main(verbosity=2)
