import numpy as np

from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.modelpacks.legacy import std2Q_XYCNOT
import pygsti.construction as pc
from pygsti import tools
import pygsti.objects.evaltree as et
from pygsti.objects import MapEvalTree, MatrixEvalTree


class EvalTreeBase(object):
    @classmethod
    def setUpClass(cls):
        opLabels = list(cls.target_model.operations.keys())
        strs = pc.create_lsgst_circuits(
            opLabels, cls.prepStrs, cls.measStrs, cls.germs, cls.maxLens,
            include_lgst=False
        )
        tools.remove_duplicates_in_place(strs)
        ret = cls.target_model.simplify_circuits(strs)
        cls.compiled_gatestrings, cls.lookup, _, _ = ret

        # Tree instances can be copied, so instantiate once and copy for each test
        # XXX this is bad testing practice but it's quick
        cls._tree = cls.constructor()
        cls._tree.initialize(cls.compiled_gatestrings)

    def setUp(self):
        self.tree = self._tree.copy()

    def test_num_final_strings(self):
        nStrs = self.tree.num_final_circuits()
        self.assertEqual(nStrs, len(self.compiled_gatestrings))

    def test_final_slice(self):
        nStrs = self.tree.num_final_circuits()
        self.assertEqual(self.tree.final_slice(None), slice(0, nStrs))  # trivial since t is not split

    def test_num_final_elements(self):
        n = self.tree.num_final_elements()
        # TODO assert correctness

    def test_evaluation_order(self):
        order = self.tree.evaluation_order()
        for k in order:
            assert(len(self.tree[k]) in (2, 3))
            # TODO does this assert correctness?

    def test_get_sub_trees(self):
        subtrees = self.tree.sub_trees()
        # TODO assert correctness

    def test_permute(self):
        # TODO no randomness
        gsl = self.tree.compute_circuits()
        dummy = np.random.rand(len(gsl))
        # TODO assert correctness of intermediate value
        dummy_computational = self.tree._permute_original_to_computation(dummy)
        dummy2 = self.tree.permute_computation_to_original(dummy_computational)
        self.assertArraysAlmostEqual(dummy, dummy2)

    def test_split_on_num_subtrees(self):
        # TODO can this be broken up?
        # Split using num_sub_trees
        gsl1 = self.tree.compute_circuits()
        lookup2 = self.tree.split(self.lookup, num_sub_trees=5)
        # TODO assert correctness
        gsl2 = self.tree.compute_circuits()
        self.assertEqual(gsl1, gsl2)

        unpermuted_list = self.tree.compute_circuits(permute=False)
        self.assertTrue(self.tree.is_split())

        subtrees = self.tree.sub_trees()
        for i, st in enumerate(subtrees):
            fslc = st.final_slice(self.tree)
            # permute=False not necessary though, since subtree is not split it's elements are not permuted
            sub_gsl = st.compute_circuits(permute=False)
            self.assertEqual(sub_gsl, unpermuted_list[fslc])

    def test_split_on_max_subtree_size(self):
        # TODO can this be broken up?
        # TODO optimize!
        # Split using max_sub_tree_size
        maxSize = 25

        gsl1 = self.tree.compute_circuits()
        lookup2 = self.tree.split(self.lookup, max_sub_tree_size=maxSize)
        # TODO assert correctness
        gsl2 = self.tree.compute_circuits()
        self.assertEqual(gsl1, gsl2)

        unpermuted_list = self.tree.compute_circuits(permute=False)

        self.assertTrue(self.tree.is_split())

        subtrees2 = self.tree.sub_trees()
        for i, st in enumerate(subtrees2):
            fslc = st.final_slice(self.tree)
            # permute=False not necessary though, since subtree is not split it's elements are not permuted
            sub_gsl = st.compute_circuits(permute=False)
            self.assertEqual(sub_gsl, unpermuted_list[fslc])

    def test_split_raises_on_conflicting_args(self):
        with self.assertRaises(ValueError):
            self.tree.split(self.lookup, max_sub_tree_size=10, num_sub_trees=10)  # can't specify both

    def test_split_raises_on_bad_num_subtrees(self):
        with self.assertRaises(ValueError):
            self.tree.split(self.lookup, num_sub_trees=0)  # num_sub_trees must be > 0


class EvalTree1QBase(EvalTreeBase):
    @classmethod
    def setUpClass(cls):
        cls.target_model = std1Q_XY.target_model()
        cls.prepStrs = std1Q_XY.fiducials
        cls.measStrs = std1Q_XY.fiducials
        cls.germs = std1Q_XY.germs
        cls.maxLens = [1, 4]
        super(EvalTree1QBase, cls).setUpClass()

    def test_construct_with_duplicate_strings(self):
        strs_with_dups = [
            (),
            ('Gx',),
            ('Gy',),
            ('Gx', 'Gy'),
            ('Gx', 'Gy'),
            (),
            ('Gx', 'Gx', 'Gx'),
            ('Gx', 'Gy', 'Gx')
        ]
        compiled_gatestrings2, lookup2, outcome_lookup2, nEls2 = self.target_model.simplify_circuits(strs_with_dups)
        tdup = self.constructor()
        tdup.initialize(compiled_gatestrings2)
        # TODO assert correctness


class EvalTree2QBase(EvalTreeBase):
    @classmethod
    def setUpClass(cls):
        cls.target_model = std2Q_XYCNOT.target_model()
        cls.prepStrs = std2Q_XYCNOT.prepStrs
        cls.measStrs = std2Q_XYCNOT.effectStrs
        cls.germs = std2Q_XYCNOT.germs
        cls.maxLens = [1, 2, 4]
        super(EvalTree2QBase, cls).setUpClass()


class MapEvalTreeBase(object):
    constructor = MapEvalTree

    def test_num_applies(self):
        ops = 0
        for iStart, remainder, iCache in self.tree:
            ops += len(remainder)
        self.assertEqual(ops, self.tree.num_applies())

    def test_squeeze(self):
        # TODO optimize!!
        self.tree.squeeze(2)
        self.tree.squeeze(0)  # special case
        # TODO assert correctness


class MatrixEvalTreeBase(object):
    constructor = MatrixEvalTree

    def test_get_min_tree_size(self):
        self.tree._min_tree_size()
        # TODO assert correctness


class MapEvalTree1QTester(MapEvalTreeBase, EvalTree1QBase, BaseCase):
    pass


class MapEvalTree2QTester(MapEvalTreeBase, EvalTree2QBase, BaseCase):
    pass


class MatrixEvalTree1QTester(MatrixEvalTreeBase, EvalTree1QBase, BaseCase):
    pass


class MatrixEvalTree2QTester(MatrixEvalTreeBase, EvalTree2QBase, BaseCase):
    pass
