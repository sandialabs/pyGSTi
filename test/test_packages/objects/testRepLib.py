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

# This class is for unifying some models that get used in this file and in testGateSets2.py
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
        grep = replib.DMOpRep_Dense(g)

        staterep2 = grep.acton(staterep)
        self.assertEqual(type(staterep2), replib.DMStateRep)

    def testRepLib_map(self):
        #Just test a Model with a "map" simtype to exercise the replib
        mdl = std.target_model()
        mdl.set_simtype("map")


        probs = mdl.probs(('Gx','Gx'))
        self.assertAlmostEqual(probs['0'], 0.0)
        self.assertAlmostEqual(probs['1'], 1.0)        

        probs2 = mdl.bulk_probs([('Gx',),('Gx','Gx'),('Gx','Gx','Gy')])
        self.assertAlmostEqual(probs2[('Gx',)]['0'], 0.5)
        self.assertAlmostEqual(probs2[('Gx',)]['1'], 0.5)        
        self.assertAlmostEqual(probs2[('Gx','Gx')]['0'], 0.0)
        self.assertAlmostEqual(probs2[('Gx','Gx')]['1'], 1.0)        
        self.assertAlmostEqual(probs2[('Gx','Gx','Gy')]['0'], 0.5)
        self.assertAlmostEqual(probs2[('Gx','Gx','Gy')]['1'], 0.5)        

        #LATER: save & check outputs of dprobs
        dprobs = mdl.bulk_dprobs([('Gx',),('Gx','Gx'),('Gx','Gx','Gy')])

        #RUN TO SAVE outputs
        if os.environ.get('PYGSTI_REGEN_REF_FILES','no').lower() in ("yes","1","true","v2"): # "v2" to only gen version-dep files
            pickle.dump(dprobs, open(compare_files + "/repLib_dprobs%s.pkl" % self.versionsuffix,'wb'))

        compare = pickle.load(open(compare_files + "/repLib_dprobs%s.pkl" % self.versionsuffix,'rb'))
        for opstr in dprobs:
            for outcomeLbl in dprobs[opstr]:
                self.assertArraysAlmostEqual(dprobs[opstr][outcomeLbl], compare[opstr][outcomeLbl])
