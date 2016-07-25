import unittest
import np

class BaseTestCase(unittest.TestCase):

    def moveTo(self, directoryName):
        self.oldwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(filename)))

    def tearDown(self):
        os.chdir(self.old)

    def assertArraysAlmostEqual(self,a,b):
        self.assertAlmostEqual( np.linalg.norm(a-b), 0 )
