import unittest
import pygsti
import numpy as np
import warnings
import pickle

from pygsti.modelpacks.legacy import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references

try:
    from pygsti.objects import fastreplib as replib
except ImportError:
    from pygsti.objects import replib


# This class is for unifying some models that get used in this file and in testGateSets2.py
class RepLibTestCase(BaseTestCase):
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
        if regenerate_references():
            pickle.dump(dprobs, open(compare_files + "/repLib_dprobs.pkl", 'wb'))

        compare = pickle.load(open(compare_files + "/repLib_dprobs.pkl", 'rb'))
        for opstr in dprobs:
            for outcomeLbl in dprobs[opstr]:
                self.assertArraysAlmostEqual(dprobs[opstr][outcomeLbl], compare[opstr][outcomeLbl])
