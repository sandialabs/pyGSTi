import numpy as np

from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.objects.modelfunction as mf


class ModelFunctionBase(object):
    def setUp(self):
        self.model = std.target_model()


class ModelFunctionUtilTester(ModelFunctionBase, BaseCase):
    def test_vecsfn_factory(self):
        # XXX is this a good test case?  EGN: seems good to me.
        def vec_dummy(vecA, vecB, mxBasis):
            return np.linalg.norm(vecA - vecB)

        Vec_dummy = mf.vecsfn_factory(vec_dummy)
        test = Vec_dummy(self.model, self.model, "Mdefault:0", "effect")
        # TODO assert correctness

class ModelFunctionInstanceTester(ModelFunctionBase, BaseCase):
    def test_with_all_dependencies(self):
        raw_gsf = mf.ModelFunction(self.model, "all")
        self.assertTrue(raw_gsf.evaluate(self.model) is None)
