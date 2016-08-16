from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.objects.spamspec import SpamSpec
import pygsti
import unittest

class TestSpamSpec(BaseTestCase):

    def setUp(self):
        super(TestSpamSpec, self).setUp()

    def test_equality(self):
        a = SpamSpec('rho0', 'Gi')
        b = SpamSpec('rho0', 'Gi')
        self.assertEqual(a, b)
        c = 'asfajsldf'
        self.assertFalse(a == c)
        c = SpamSpec('E0', 'Gi')
        self.assertFalse(a == c)
        print(a)

if __name__ == '__main__':
    unittest.main(verbosity=2)
