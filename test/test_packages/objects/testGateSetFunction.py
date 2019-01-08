import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os
import copy

from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some models that get used in this file and in testGateSets2.py
class ModelFunctionTestCase(BaseTestCase):

    def setUp(self):
        super(ModelFunctionTestCase, self).setUp()


    def test_modelfunction(self):
        mdl = std.target_model()
        raw_gsf = pygsti.objects.modelfunction.ModelFunction(mdl, "all")
        self.assertTrue(raw_gsf.evaluate(mdl) is None)

        #another case that isn't covered elsewhere: "effect" mode of a vec-fn
        def vec_dummy(vecA, vecB, mxBasis):
            return np.linalg.norm(vecA-vecB)
        Vec_dummy = pygsti.objects.modelfunction.vecsfn_factory(vec_dummy)
          # init args == (model1, model2, label, typ)
        test = Vec_dummy(mdl, mdl, "Mdefault:0", "effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
