import unittest
import pygsti
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class KCoverageTestCase(BaseTestCase):

    def setUp(self):
        super(KCoverageTestCase, self).setUp()

    def test_kcoverage(self):
        n=10 # nqubits
        k=4  # number of "labels" needing distribution
        rows = pygsti.construction.get_kcoverage_template(n,k, verbosity=2) 
        pygsti.construction.check_kcoverage_template(rows,n,k, verbosity=1) #asserts success
