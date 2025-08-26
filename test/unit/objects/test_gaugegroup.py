import numpy as np

from pygsti.modelmembers import operations as op
from pygsti.models import gaugegroup as ggrp
from pygsti.baseobjs.statespace import QubitSpace, ExplicitStateSpace
from ..util import BaseCase


class GaugeGroupBase(object):

    HAS_DERIV_WRT_PARAMS = True

    def setUp(self):
        self.state_space = QubitSpace(1)
        self.rng = np.random.default_rng(0)
    
    def test_construction(self):
        params = self.gg.initial_params
        self.assertEqual(len(params), self.n_params)
        self.assertEqual(self.gg.num_params, self.n_params)
        element = self.gg.compute_element(params)
        # XXX is this necessary?  EGN: maybe not, but it asserts correctness and should be fast
        self.assertIsInstance(element, self.element_type)

    def test_element_construction(self):
        el = self.gg.compute_element(self.gg.initial_params)
        self.assertEqual(el.num_params, self.n_params)

    def test_element_get_transform_matrix(self):
        el = self.gg.compute_element(self.gg.initial_params)
        mx = el.transform_matrix
        # TODO assert correctness

    def test_element_get_transform_matrix_inverse(self):
        el = self.gg.compute_element(self.gg.initial_params)
        mx = el.transform_matrix
        inv = el.transform_matrix_inverse
        self.assertArraysAlmostEqual(np.linalg.inv(mx), inv)

    def test_element_deriv_wrt_params(self):
        if self.HAS_DERIV_WRT_PARAMS:
            el = self.gg.compute_element(self.gg.initial_params)
            deriv = el.deriv_wrt_params()
            # TODO assert correctness

    def test_element_to_from_vector(self):
        el = self.gg.compute_element(self.gg.initial_params)
        v0 = el.to_vector().copy()
        m0 = el.transform_matrix.copy()
        num_params = v0.size
        if num_params > 0:
            v1 = self.rng.random(size=(num_params,))
            el.from_vector(v1)
            m1 = el.transform_matrix.copy()
            self.assertGreater(np.linalg.norm(m1 - m0), 0.0)
            el.from_vector(v0)
            m2 = el.transform_matrix.copy()
            self.assertArraysAlmostEqual(m0, m2)
        else:
            # we just check that from_vector raises no error when provided 
            # with a vector of length zero.
            el.from_vector(v0)
        return


class GaugeGroupTester(GaugeGroupBase, BaseCase):
    # XXX do we need coverage of an abstract base class?
    # XXX should this class even be instantiatable?  EGN: no, it's just a base class.
    n_params = 0
    element_type = ggrp.GaugeGroupElement

    def setUp(self):
        self.gg = ggrp.GaugeGroup('myGaugeGroupName')

    def test_element_get_transform_matrix_inverse(self):
        el = self.gg.compute_element(self.gg.initial_params)
        inv = el.transform_matrix_inverse
        self.assertIsNone(inv)

    def test_element_to_from_vector(self):
        pass  # abstract


class OpGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 16
    element_type = ggrp.OpGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.OpGaugeGroup(op.FullArbitraryOp(np.identity(4, 'd'), state_space=self.state_space),
                                    ggrp.OpGaugeGroupElement, 'myGateGaugeGroupName')


class FullGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 16
    element_type = ggrp.FullGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.FullGaugeGroup(self.state_space)


class TPGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 12
    element_type = ggrp.TPGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TPGaugeGroup(self.state_space)


class DiagGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 4
    element_type = ggrp.DiagGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.DiagGaugeGroup(self.state_space)


class TPDiagGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 3
    element_type = ggrp.TPDiagGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TPDiagGaugeGroup(self.state_space)


class SpamGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 2
    element_type = ggrp.SpamGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.SpamGaugeGroup(self.state_space)


class TrivialGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 0
    element_type = ggrp.TrivialGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TrivialGaugeGroup(self.state_space)


class DirectSumGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 3
    element_type = ggrp.DirectSumUnitaryGroupElement
    HAS_DERIV_WRT_PARAMS = False

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.state_space = ExplicitStateSpace(['dummy'],[5])
        g1 = ggrp.TrivialGaugeGroup(ExplicitStateSpace(['T0']))
        g2 = ggrp.UnitaryGaugeGroup(QubitSpace(1), 'pp')
        self.gg = ggrp.DirectSumUnitaryGroup((g1, g2), 'std')
