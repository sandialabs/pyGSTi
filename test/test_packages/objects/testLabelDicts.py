import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class LabelDictTestCase(BaseTestCase):

    def setUp(self):
        super(LabelDictTestCase, self).setUp()

    def testOutcomeLabelDict(self):
        d = pygsti.objects.labeldicts.OutcomeLabelDict([( ('0',), 90 ), ( ('1',), 10)])
        self.assertEqual(d['0'], 90) #don't need tuple when they're 1-tuples
        self.assertEqual(d['1'], 10) #don't need tuple when they're 1-tuples

        s = pickle.dumps(d)
        d2 = pickle.loads(s)

        d3 = d.copy()
        
