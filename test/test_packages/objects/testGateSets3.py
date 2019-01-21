import unittest
import pygsti
import numpy as np
import warnings
import os

from .testGateSets import GateSetTestCase
#from pygsti.objects.mapforwardsim import MapForwardSimulator

class TestGateSetMethods(GateSetTestCase):

    def setUp(self):
        super(TestGateSetMethods, self).setUp()

    def test_hproduct(self):
        self.model.hproduct(('Gx', 'Gi'))

    def test_tp_dist(self):
        self.assertAlmostEqual(self.tp_gateset.tpdist(), 3.52633900335e-16, 5)

    def test_strdiff(self):
        self.model.strdiff(self.tp_gateset)

    def test_bad_dimm(self):
        copiedGateset = self.model.copy()
        copiedGateset = copiedGateset.increase_dimension(11)
        with self.assertRaises(AssertionError):
            copiedGateset.rotate((0.1,0.1,0.1))
        with self.assertRaises(AssertionError):
            copiedGateset.randomize_with_unitary(1, randState=np.random.RandomState()) # scale shouldn't matter

    def test_mem_estimates(self):

        model = pygsti.construction.build_explicit_model(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])

        mgateset = self.model.copy()
        #mgateset._calcClass = MapForwardSimulator
        mgateset.set_simtype('map')

        est = model._fwdsim().estimate_mem_usage(["bulk_fill_probs","bulk_fill_dprobs","bulk_fill_hprobs"],
                                                 cache_size=100, num_subtrees=2, 
                                                 num_subtree_proc_groups=1, num_param1_groups=1, 
                                                 num_param2_groups=1, num_final_strs=100)

        est = mgateset._fwdsim().estimate_mem_usage(["bulk_fill_probs","bulk_fill_dprobs","bulk_fill_hprobs"],
                                                 cache_size=100, num_subtrees=2, 
                                                 num_subtree_proc_groups=1, num_param1_groups=1, 
                                                  num_param2_groups=1, num_final_strs=100)
