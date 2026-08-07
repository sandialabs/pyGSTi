import unittest
import numpy as np
from pygsti.data.hypothesistest import HypothesisTest
from ..util import BaseCase


class HypothesisTestTester(BaseCase):
    def test_flat_holm_step_down(self):
        # 1a. Flat (non-nested) weighted Holm step-down
        h = HypothesisTest(['a', 'b', 'c'], significance=0.05)
        h.add_pvalues({'a': 0.001, 'b': 0.02, 'c': 0.5})
        h.run()

        # a is tested at significance 0.05 / 3 = 0.016667
        # Since p_a = 0.001 <= 0.016667, a is rejected.
        # Its significance is re-allocated to b and c.
        # Remaining nulls: b, c. dynamic_local_significance:
        # b: 0.05/3 + (0.05/3)*(0.5) = 0.025
        # c: 0.05/3 + (0.05/3)*(0.5) = 0.025
        # b is tested at 0.025. Since p_b = 0.02 <= 0.025, b is rejected.
        # Its significance is re-allocated to c.
        # Remaining nulls: c. dynamic_local_significance:
        # c: 0.025 + 0.025*1 = 0.05.
        # c is tested at 0.05. Since p_c = 0.5 > 0.05, c is not rejected.
        self.assertTrue(h.hypothesis_rejected['a'])
        self.assertTrue(h.hypothesis_rejected['b'])
        self.assertFalse(h.hypothesis_rejected['c'])

        self.assertAlmostEqual(h.significance_tested_at['a'], 0.05 / 3)
        self.assertAlmostEqual(h.significance_tested_at['b'], 0.025)
        self.assertAlmostEqual(h.significance_tested_at['c'], 0.05)

        self.assertAlmostEqual(h.pvalue_pseudothreshold['a'], 0.05 / 3)
        self.assertAlmostEqual(h.pvalue_pseudothreshold['b'], 0.025)
        self.assertAlmostEqual(h.pvalue_pseudothreshold['c'], 0.05)

    def test_flat_holm_none_rejected(self):
        h = HypothesisTest(['a', 'b', 'c'], significance=0.05)
        h.add_pvalues({'a': 0.9, 'b': 0.9, 'c': 0.9})
        h.run()

        self.assertFalse(h.hypothesis_rejected['a'])
        self.assertFalse(h.hypothesis_rejected['b'])
        self.assertFalse(h.hypothesis_rejected['c'])

        self.assertAlmostEqual(h.significance_tested_at['a'], 0.05 / 3)
        self.assertAlmostEqual(h.significance_tested_at['b'], 0.05 / 3)
        self.assertAlmostEqual(h.significance_tested_at['c'], 0.05 / 3)

    def test_custom_weighting(self):
        # 1b. Custom weighting dict
        # Normalized weights: a = 3/4, b = 1/4
        # At alpha=0.05: local alpha_a = 0.0375, alpha_b = 0.0125
        h = HypothesisTest(['a', 'b'], significance=0.05, weighting={'a': 3, 'b': 1})
        h.add_pvalues({'a': 0.03, 'b': 0.03})
        h.run()

        # a is tested at 0.0375. Since 0.03 <= 0.0375, a is rejected.
        # Its significance is re-allocated to b.
        # Remaining: b. b gets all significance = 0.05.
        # b is tested at 0.05. Since 0.03 <= 0.05, b is also rejected.
        self.assertTrue(h.hypothesis_rejected['a'])
        self.assertTrue(h.hypothesis_rejected['b'])

        self.assertAlmostEqual(h.local_significance['a'], 0.0375)
        self.assertAlmostEqual(h.local_significance['b'], 0.0125)

    def test_local_corrections_parameterization_group1_hochberg(self):
        # 1c. Hochberg correction when aggregate rejected
        # agg gets weight 0.5, group ('x', 'y', 'z') gets 0.5. alpha = 0.05.
        # local alphas = 0.025 each.
        # p_agg = 0.001 (rejected). Since agg rejected, its 0.025 is passed to ('x', 'y', 'z').
        # So ('x', 'y', 'z') is tested with alpha_group = 0.05.
        # For Hochberg step-up with n=3, alpha=0.05, and p_sorted = [0.9, 0.02, 0.02] (reversed: 0.9, 0.02, 0.02):
        # - i=0 (largest p-value, 0.9): tested at 0.05 / (0 + 1) = 0.05. 0.9 <= 0.05 is False.
        # - i=1 (next, 0.02): tested at 0.05 / (1 + 1) = 0.025. Since 0.02 <= 0.025,
        #   all remaining/smaller p-values are rejected (i.e. x and y).
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='Hochberg', significance=0.05)
        h.add_pvalues({'agg': 0.001, 'x': 0.02, 'y': 0.02, 'z': 0.9})
        h.run()

        self.assertTrue(h.hypothesis_rejected['agg'])
        self.assertTrue(h.hypothesis_rejected['x'])
        self.assertTrue(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.significance_tested_at['agg'], 0.025)
        self.assertAlmostEqual(h.significance_tested_at['x'], 0.025)
        self.assertAlmostEqual(h.significance_tested_at['y'], 0.025)
        self.assertAlmostEqual(h.significance_tested_at['z'], 0.05)
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.025)

    def test_local_corrections_parameterization_group1_holms(self):
        # Holm correction when aggregate rejected. alpha_group = 0.05.
        # p = {x: 0.02, y: 0.02, z: 0.9}.
        # For step-down Holm with n=3:
        # - smallest p-value is x (or y) = 0.02. Tested at 0.05 / 3 = 0.016667.
        #   0.02 <= 0.016667 is False. No rejections!
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='Holms', significance=0.05)
        h.add_pvalues({'agg': 0.001, 'x': 0.02, 'y': 0.02, 'z': 0.9})
        h.run()

        self.assertTrue(h.hypothesis_rejected['agg'])
        self.assertFalse(h.hypothesis_rejected['x'])
        self.assertFalse(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.016666666666666666)

    def test_local_corrections_parameterization_group2_bonferroni(self):
        # Aggregate not rejected. alpha_group remains 0.025.
        # Bonferroni with n=3: threshold = 0.025 / 3 = 0.008333.
        # p = {x: 0.01, y: 0.02, z: 0.9}. None of these <= 0.008333, so none rejected.
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='Bonferroni', significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 0.01, 'y': 0.02, 'z': 0.9})
        h.run()

        self.assertFalse(h.hypothesis_rejected['agg'])
        self.assertFalse(h.hypothesis_rejected['x'])
        self.assertFalse(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.significance_tested_at['x'], 0.008333333333333333)
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.008333333333333333)

    def test_local_corrections_parameterization_group2_benjamini_hochberg(self):
        # Aggregate not rejected. alpha_group remains 0.025.
        # BH with n=3, p = {x: 0.01, y: 0.02, z: 0.9}.
        # Re-ordered p-values (largest first):
        # - i=0 (p_z = 0.9): tested at 0.025 * 3 / 3 = 0.025. 0.9 <= 0.025 is False.
        # - i=1 (p_y = 0.02): tested at 0.025 * 2 / 3 = 0.016667. 0.02 <= 0.016667 is False.
        # - i=2 (p_x = 0.01): tested at 0.025 * 1 / 3 = 0.008333. 0.01 <= 0.008333 is False.
        # None rejected.
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='Benjamini-Hochberg', significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 0.01, 'y': 0.02, 'z': 0.9})
        h.run()

        self.assertFalse(h.hypothesis_rejected['agg'])
        self.assertFalse(h.hypothesis_rejected['x'])
        self.assertFalse(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.significance_tested_at['x'], 0.008333333333333333)
        self.assertAlmostEqual(h.significance_tested_at['y'], 0.016666666666666666)
        self.assertAlmostEqual(h.significance_tested_at['z'], 0.025)
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.008333333333333333)

    def test_local_corrections_parameterization_group2_benjamini_hochberg_rejections(self):
        # BH with n=3, alpha=0.025, p = {x: 0.01, y: 0.015, z: 0.9}.
        # - i=0 (p_z = 0.9): tested at 0.025 * 3 / 3 = 0.025. False.
        # - i=1 (p_y = 0.015): tested at 0.025 * 2 / 3 = 0.016667. Since 0.015 <= 0.016667,
        #   all remaining/smaller p-values (x and y) are rejected!
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='Benjamini-Hochberg', significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 0.01, 'y': 0.015, 'z': 0.9})
        h.run()

        self.assertTrue(h.hypothesis_rejected['x'])
        self.assertTrue(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.016666666666666666)

    def test_local_corrections_parameterization_group2_none(self):
        # Aggregate not rejected. alpha_group remains 0.025.
        # correction='none': each tested at 0.025.
        # p = {x: 0.01, y: 0.02, z: 0.9}. x and y rejected because 0.01 <= 0.025 and 0.02 <= 0.025.
        h = HypothesisTest(['agg', ('x', 'y', 'z')], weighting={'agg': 0.5, ('x', 'y', 'z'): 0.5},
                           local_corrections='none', significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 0.01, 'y': 0.02, 'z': 0.9})
        h.run()

        self.assertFalse(h.hypothesis_rejected['agg'])
        self.assertTrue(h.hypothesis_rejected['x'])
        self.assertTrue(h.hypothesis_rejected['y'])
        self.assertFalse(h.hypothesis_rejected['z'])
        self.assertAlmostEqual(h.significance_tested_at['x'], 0.025)
        self.assertAlmostEqual(h.pvalue_pseudothreshold[('x', 'y', 'z')], 0.025)

    def test_error_paths(self):
        # 1d. Error paths and validation
        # Bad significance bounds
        with self.assertRaises(AssertionError):
            HypothesisTest(['a', 'b'], significance=0.0)
        with self.assertRaises(AssertionError):
            HypothesisTest(['a', 'b'], significance=1.0)

        # Bad local_corrections string value
        with self.assertRaises(AssertionError):
            HypothesisTest(['a', 'b'], local_corrections='bogus')

        # run() without p-values
        h = HypothesisTest(['a', 'b'])
        with self.assertRaises(AssertionError):
            h.run()

        # ValueError raised inside nested correction dispatch for bad dict value
        h = HypothesisTest(['agg', ('x', 'y')], weighting={'agg': 0.5, ('x', 'y'): 0.5},
                           local_corrections={('x', 'y'): 'bogus'}, significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 0.5, 'y': 0.5})
        with self.assertRaises(ValueError):
            h.run()

    def test_pvalue_copy_contract(self):
        # add_pvalues must copy rather than reference
        p_dict = {'a': 0.1, 'b': 0.2}
        h = HypothesisTest(['a', 'b'])
        h.add_pvalues(p_dict)
        p_dict['a'] = 0.9  # mutate the source dict
        self.assertEqual(h.pvalues['a'], 0.1)

    @unittest.expectedFailure
    def test_bug_passing_graph_numpy_array(self):
        # Bug 1: passing_graph as an array is unimplemented.
        # Should build and run successfully without AttributeError on self.passing_graph.
        g = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = HypothesisTest(['a', 'b'], passing_graph=g, significance=0.05)
        h.add_pvalues({'a': 0.001, 'b': 0.001})
        h.run()
        self.assertTrue(h.hypothesis_rejected['a'])
        self.assertTrue(h.hypothesis_rejected['b'])

    @unittest.expectedFailure
    def test_bug_passing_graph_none(self):
        # Bug 2: passing_graph='none' fails assertion that it must be 'Holms'.
        # This is passed by DataComparator when pass_alpha=False.
        h = HypothesisTest(['a', 'b'], passing_graph='none')
        self.assertEqual(h.passing_graph, 'none')

    @unittest.expectedFailure
    def test_bug_nested_holms_all_rejected_zerodivision(self):
        # Bug 3: ZeroDivisionError when nested Holms rejects everything.
        # When all hypotheses in a nested group are rejected, len(dynamic_hypotheses) becomes 0.
        # Line 384 then does `significance / 0`, raising ZeroDivisionError.
        h = HypothesisTest(['agg', ('x', 'y')], local_corrections='Holms', significance=0.05)
        h.add_pvalues({'agg': 0.9, 'x': 1e-9, 'y': 1e-9})
        h.run()
        self.assertTrue(h.hypothesis_rejected['x'])
        self.assertTrue(h.hypothesis_rejected['y'])

    @unittest.expectedFailure
    def test_bug_nested_list_unhashable(self):
        # Bug 4: List-valued nested hypotheses cause unhashable type error.
        # __init__ accepts list-valued hypotheses but later tries to use them as dict keys.
        h = HypothesisTest(['agg', ['x', 'y']])
        self.assertTrue(h.nested_hypotheses[['x', 'y']])

    @unittest.expectedFailure
    def test_bug_single_hypothesis_divide_by_zero_warning(self):
        # Bug 5: Single hypothesis raises divide-by-zero RuntimeWarning from numpy:
        # self.passing_graph[hind, :] = _np.ones(...) / (len(self.hypotheses) - 1)
        # when len(self.hypotheses) == 1, (1-1) is 0, raising RuntimeWarning under pytest error filter.
        with self.assertNoWarns(RuntimeWarning):
            h = HypothesisTest(['a'], significance=0.05)
            h.add_pvalues({'a': 0.01})
            h.run()
