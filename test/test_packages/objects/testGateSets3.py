import unittest
import pygsti
import numpy as np
import warnings
import os

from .testGateSets import GateSetTestCase

class TestGateSetMethods(GateSetTestCase):

    def setUp(self):
        super(TestGateSetMethods, self).setUp()

    def test_hproduct(self):
        self.gateset.hproduct(('Gx', 'Gi'))

    def test_tp_dist(self):
        self.assertAlmostEqual(self.tp_gateset.tpdist(), 3.52633900335e-16, 5)

    def test_strdiff(self):
        self.gateset.strdiff(self.tp_gateset)

    def test_bad_dimm(self):
        copiedGateset = self.gateset.copy()
        copiedGateset = copiedGateset.increase_dimension(11)
        with self.assertRaises(AssertionError):
            copiedGateset.rotate(0.1)
        with self.assertRaises(AssertionError):
            copiedGateset.randomize_with_unitary(1, randState=np.random.RandomState()) # scale shouldn't matter
