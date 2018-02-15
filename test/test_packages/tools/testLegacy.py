import unittest
import pygsti
import os

from pygsti.tools import legacytools

class LegacyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_deprecation_warning(self):

        @legacytools.deprecated_fn("Replacement function name")
        def oldFn(x):
            return x

        oldFn(5)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
