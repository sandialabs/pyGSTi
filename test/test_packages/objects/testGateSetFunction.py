import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os
import copy

from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class GatesetFunctionTestCase(BaseTestCase):

    def setUp(self):
        super(GatesetFunctionTestCase, self).setUp()


    def test_gatesetfunction(self):
        gs = std.gs_target.copy()
        raw_gsf = pygsti.objects.gatesetfunction.GateSetFunction(gs, "all")
        self.assertTrue(raw_gsf.evaluate(gs) is None)

        #another case that isn't covered elsewhere: "effect" mode of a vec-fn
        def vec_dummy(vecA, vecB, mxBasis):
            return np.linalg.norm(vecA-vecB)
        Vec_dummy = pygsti.objects.gatesetfunction.vecsfn_factory(vec_dummy)
          # init args == (gateset1, gateset2, label, typ)
        test = Vec_dummy(gs, gs, "Mdefault:0", "effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
