import unittest
import pygsti
import numpy as np

from  pygsti.objects import gaugegroup as ggrp
from  pygsti.objects import gate
import pygsti.construction as pc

from ..testutils import BaseTestCase, compare_files, temp_files

class GaugeGroupTestCase(BaseTestCase):
    def setUp(self):
        super(GaugeGroupTestCase, self).setUp()

    def test_construction(self):
        gg   = ggrp.GaugeGroup('myGaugeGroupName')
        ggg  = ggrp.GateGaugeGroup(gate.FullyParameterizedGate(np.identity(4,'d')),
                                   ggrp.GateGaugeGroupElement,'myGateGaugeGroupName')
        fgg  = ggrp.FullGaugeGroup(4)
        tgg  = ggrp.TPGaugeGroup(4)
        dgg  = ggrp.DiagGaugeGroup(4)
        tdgg = ggrp.TPDiagGaugeGroup(4)
        sgg  = ggrp.SpamGaugeGroup(4)
        trgg = ggrp.TrivialGaugeGroup(4)

        gg_ip   = gg.get_initial_params()
        ggg_ip  = ggg.get_initial_params()
        fgg_ip  = fgg.get_initial_params()
        tgg_ip  = tgg.get_initial_params()
        dgg_ip  = dgg.get_initial_params()
        tdgg_ip = tdgg.get_initial_params()
        sgg_ip  = sgg.get_initial_params()
        trgg_ip = trgg.get_initial_params()

        self.assertEqual(len(gg_ip), 0)
        self.assertEqual(len(ggg_ip), 16)
        self.assertEqual(len(fgg_ip), 16)
        self.assertEqual(len(tgg_ip), 12)
        self.assertEqual(len(dgg_ip), 4)
        self.assertEqual(len(tdgg_ip), 3)
        self.assertEqual(len(sgg_ip), 2)
        self.assertEqual(len(trgg_ip), 0)

        self.assertEqual(gg.num_params(), 0)
        self.assertEqual(ggg.num_params(), 16)
        self.assertEqual(fgg.num_params(), 16)
        self.assertEqual(tgg.num_params(), 12)
        self.assertEqual(dgg.num_params(), 4)
        self.assertEqual(tdgg.num_params(), 3)
        self.assertEqual(sgg.num_params(), 2)
        self.assertEqual(trgg.num_params(), 0)

        gg_el   = gg.get_element(gg_ip)
        ggg_el  = ggg.get_element(ggg_ip)
        fgg_el  = fgg.get_element(fgg_ip) 
        tgg_el  = tgg.get_element(tgg_ip) 
        dgg_el  = dgg.get_element(dgg_ip) 
        tdgg_el = tdgg.get_element(tdgg_ip)
        sgg_el  = sgg.get_element(sgg_ip)
        trgg_el = trgg.get_element(trgg_ip)
        
        self.assertIsInstance(gg_el, ggrp.GaugeGroupElement)
        self.assertIsInstance(ggg_el, ggrp.GateGaugeGroupElement)
        self.assertIsInstance(fgg_el, ggrp.FullGaugeGroupElement)
        self.assertIsInstance(tgg_el, ggrp.TPGaugeGroupElement)
        self.assertIsInstance(dgg_el, ggrp.DiagGaugeGroupElement)
        self.assertIsInstance(tdgg_el, ggrp.TPDiagGaugeGroupElement)
        self.assertIsInstance(sgg_el, ggrp.SpamGaugeGroupElement)
        self.assertIsInstance(trgg_el, ggrp.TrivialGaugeGroupElement)

    def test_elements(self):
        ggs = []
        ggs.append(ggrp.GaugeGroup('myGroupName'))
        ggs.append(ggrp.GateGaugeGroup(gate.FullyParameterizedGate(np.identity(4,'d')),
                                       ggrp.GateGaugeGroupElement,'myGateGroupName'))
        ggs.append(ggrp.FullGaugeGroup(4))
        ggs.append(ggrp.TPGaugeGroup(4))
        ggs.append(ggrp.DiagGaugeGroup(4))
        ggs.append(ggrp.TPDiagGaugeGroup(4))
        ggs.append(ggrp.SpamGaugeGroup(4))
        ggs.append(ggrp.TrivialGaugeGroup(4))

        for gg in ggs:
            ip = gg.get_initial_params()
            el = gg.get_element(ip)
            nP = el.num_params()
            mx = el.get_transform_matrix()
            inv  = el.get_transform_matrix_inverse()
            invB = el.get_transform_matrix_inverse()
            deriv = el.deriv_wrt_params()
            v = el.to_vector()
            
            el2 = gg.get_element(ip)
            el2.from_vector(v)
            mx2 = el2.get_transform_matrix()
            inv2 = el2.get_transform_matrix_inverse()

            if len(v) > 0:
                self.assertAlmostEqual(np.linalg.norm(np.linalg.inv(mx)-inv),0)
                self.assertAlmostEqual(np.linalg.norm(inv-invB),0)
                self.assertAlmostEqual(np.linalg.norm(mx-mx2),0)
                self.assertAlmostEqual(np.linalg.norm(inv-inv2),0)
            

            
