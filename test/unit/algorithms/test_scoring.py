import numpy as np

from pygsti.algorithms import scoring
from ..util import BaseCase


class ScoringTester(BaseCase):
    def setUp(self):
        self.eigvals = np.array([1e-6, 1e-4, 1.0, 2.0])

    def test_list_score_all(self):
        s1 = scoring.list_score(self.eigvals, 'all')
        self.assertEqual(s1, 1010001.5)

    def test_list_score_worst(self):
        s2 = scoring.list_score(self.eigvals, 'worst')
        self.assertEqual(s2, 1000000)

    def test_list_score_raises_on_bad_score_function(self):
        with self.assertRaises(ValueError):
            scoring.list_score(self.eigvals, 'foobar')
