import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os

from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

try:
    from pygsti.objects import fastreplib as replib
except ImportError:
    from pygsti.objects import replib

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class RepLibTestCase(BaseTestCase):

    def setUp(self):
        super(RepLibTestCase, self).setUp()

    def testRepLib_basic(self):
        #just some simple tests of replib functions for now
        x = np.zeros(4,'d')
        staterep = replib.DMStateRep(x) # state rep

        x = np.zeros(4,'d')
        erep = replib.DMEffectRep_Dense(x)
        self.assertAlmostEqual(erep.probability(staterep), 0.0)

        g = np.zeros((4,4),'d')
        grep = replib.DMGateRep_Dense(g)

        staterep2 = grep.acton(staterep)
        self.assertEqual(type(staterep2), replib.DMStateRep)

    def testRepLib_map(self):
        #Just test a GateSet with a "map" simtype to exercise the replib
        gs = std.gs_target.copy()
        gs.set_simtype("map")


        probs = gs.probs(('Gx','Gx'))
        self.assertAlmostEqual(probs['0'], 0.0)
        self.assertAlmostEqual(probs['1'], 1.0)        

        probs2 = gs.bulk_probs([('Gx',),('Gx','Gx'),('Gx','Gx','Gy')])
        self.assertAlmostEqual(probs2[('Gx',)]['0'], 0.5)
        self.assertAlmostEqual(probs2[('Gx',)]['1'], 0.5)        
        self.assertAlmostEqual(probs2[('Gx','Gx')]['0'], 0.0)
        self.assertAlmostEqual(probs2[('Gx','Gx')]['1'], 1.0)        
        self.assertAlmostEqual(probs2[('Gx','Gx','Gy')]['0'], 0.5)
        self.assertAlmostEqual(probs2[('Gx','Gx','Gy')]['1'], 0.5)        

        #LATER: save & check outputs of dprobs
        dprobs = gs.bulk_dprobs([('Gx',),('Gx','Gx'),('Gx','Gx','Gy')])

        #RUN TO save outputs
        #pickle.dump(dprobs, open(compare_files + "/repLib_dprobs%s.pkl" % self.versionsuffix,'wb'))

        compare = pickle.load(open(compare_files + "/repLib_dprobs%s.pkl" % self.versionsuffix,'rb'))
        for gstr in dprobs:
            for outcomeLbl in dprobs[gstr]:
                self.assertArraysAlmostEqual(dprobs[gstr][outcomeLbl], compare[gstr][outcomeLbl])


    def test_CNOT_convention(self):
        #TODO: move elsewhere?
        
        #1-off check (unrelated to fast acton) - showing that CNOT gate convention is CNOT(control,target)
        # so for CNOT:1:2 gates, 1 is the *control* and 2 is the *target*
        from pygsti.construction import std2Q_XYICNOT
        std_cnot = pygsti.tools.process_mx_to_unitary(pygsti.tools.change_basis(std2Q_XYICNOT.gs_target.gates['Gcnot'],'pp','std'))
        state_10 = pygsti.tools.dmvec_to_state(pygsti.tools.change_basis(std2Q_XYICNOT.gs_target.povms['Mdefault']['10'],"pp","std"))

        # if first qubit is control, CNOT should leave 00 & 01 (first 2 rows/cols) alone:
        expected_cnot = np.array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                                  [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                                  [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                                  [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

        # first qubit is most significant, so rows are 00,01,10,11
        expected_state10 = np.array([[0.+0.j],
                                     [0.+0.j],
                                     [1.+0.j],
                                     [0.+0.j]]) 
        
        self.assertArraysAlmostEqual(std_cnot, expected_cnot)
        self.assertArraysAlmostEqual(state_10, expected_state10)
