import unittest
import numpy as np
import os

class ToolsTestCase(unittest.TestCase):

    def setUp(self):
        # move working directories
        self.old = os.getcwd()
        # This will result in the same directory, even though when another module calls this, file points to toolsBaseCase.py
        os.chdir(os.path.abspath(os.path.dirname(__file__))) 

        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

    def tearDown(self):
        os.chdir(self.old)

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )
