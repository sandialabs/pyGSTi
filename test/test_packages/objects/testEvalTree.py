import unittest
import pygsti
import numpy as np

import pygsti
from ..testutils import BaseTestCase, compare_files, temp_files

from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.modelpacks.legacy import std2Q_XYCNOT
from pygsti.objects.mapforwardsim import MapForwardSimulator

class EvalTreeTestCase(BaseTestCase):

    def setUp(self):
        super(EvalTreeTestCase, self).setUp()

    def test_base_tree(self):
        raw_tree = pygsti.obj.EvalTree()
        with self.assertRaises(NotImplementedError):
            raw_tree.initialize(None)
        with self.assertRaises(NotImplementedError):
            raw_tree.generate_circuit_list()
        with self.assertRaises(NotImplementedError):
            raw_tree.split(None)





    def test_mapevaltree(self):
        # An additional specific test added from debugging mapevaltree splitting
        mgateset = pygsti.construction.build_explicit_model(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        mgateset._calcClass = MapForwardSimulator

        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')
        gatestring3 = ('Gx',)
        gatestring4 = ('Gy','Gy')
        #mevt,mlookup,moutcome_lookup = mgateset.bulk_evaltree( [gatestring1,gatestring2] )
        #mevt,mlookup,moutcome_lookup = mgateset.bulk_evaltree( [gatestring1,gatestring4] )
        mevt,mlookup,moutcome_lookup = mgateset.bulk_evaltree( [gatestring1,gatestring2,gatestring3,gatestring4] )
        print("Tree = ",mevt)
        print("Cache size = ",mevt.cache_size())
        print("lookup = ",mlookup)
        print()

        self.assertEqual(mevt[:], [(0, ('Gy',), 1),
                                   (1, ('Gy',), None),
                                   (None, ('rho0', 'Gx',), 0),
                                   (None, ('rho0', 'Gy', 'Gy'), None)])
        self.assertEqual(mevt.cache_size(),2)
        self.assertEqual(mevt.get_evaluation_order(),[2, 0, 1, 3])
        self.assertEqual(mevt.num_final_strings(),4)

         ## COPY
        mevt_copy = mevt.copy()
        print("Tree copy = ",mevt_copy)
        print("Cache size = ",mevt_copy.cache_size())
        print("Eval order = ",mevt_copy.get_evaluation_order())
        print("Num final = ",mevt_copy.num_final_strings())
        print()

        self.assertEqual(mevt_copy[:], [(0, ('Gy',), 1),
                                   (1, ('Gy',), None),
                                   (None, ('rho0', 'Gx',), 0),
                                   (None, ('rho0', 'Gy', 'Gy'), None)])
        self.assertEqual(mevt_copy.cache_size(),2)
        self.assertEqual(mevt_copy.get_evaluation_order(),[2, 0, 1, 3])
        self.assertEqual(mevt_copy.num_final_strings(),4)

          ## SQUEEZE
        maxCacheSize = 1
        mevt_squeeze = mevt.copy()
        mevt_squeeze.squeeze(maxCacheSize)
        print("Squeezed Tree = ",mevt_squeeze)
        print("Cache size = ",mevt_squeeze.cache_size())
        print("Eval order = ",mevt_squeeze.get_evaluation_order())
        print("Num final = ",mevt_squeeze.num_final_strings())
        print()

        self.assertEqual(mevt_squeeze[:], [(0, ('Gy',), None),
                                           (0, ('Gy','Gy'), None),
                                           (None, ('rho0', 'Gx',), 0),
                                           (None, ('rho0', 'Gy', 'Gy'), None)])

        self.assertEqual(mevt_squeeze.cache_size(),maxCacheSize)
        self.assertEqual(mevt_squeeze.get_evaluation_order(),[2, 0, 1, 3])
        self.assertEqual(mevt_squeeze.num_final_strings(),4)

          #SPLIT
        mevt_split = mevt.copy()
        mlookup_splt = mevt_split.split(mlookup,num_sub_trees=4)
        print("Split tree = ",mevt_split)
        print("new lookup = ",mlookup_splt)
        print()

        self.assertEqual(mevt_split[:], [(None, ('rho0', 'Gx',), 0),
                                         (0, ('Gy',), 1),
                                         (1, ('Gy',), None),
                                         (None, ('rho0', 'Gy', 'Gy'), None)])
        self.assertEqual(mevt_split.cache_size(),2)
        self.assertEqual(mevt_split.get_evaluation_order(),[0, 1, 2, 3])
        self.assertEqual(mevt_split.num_final_strings(),4)


        subtrees = mevt_split.get_sub_trees()
        print("%d subtrees" % len(subtrees))
        self.assertEqual(len(subtrees),4)
        for i,subtree in enumerate(subtrees):
            print("Sub tree %d = " % i,subtree,
                  " csize = ",subtree.cache_size(),
                  " eval = ",subtree.get_evaluation_order(),
                  " nfinal = ",subtree.num_final_strings())
            self.assertEqual(subtree.cache_size(),0)
            self.assertEqual(subtree.get_evaluation_order(),[0])
            self.assertEqual(subtree.num_final_strings(),1)

        probs = np.zeros( mevt.num_final_elements(), 'd')
        mgateset.bulk_fill_probs(probs, mevt)
        print("probs = ",probs)
        print("lookup = ",mlookup)
        self.assertArraysAlmostEqual(probs, np.array([ 0.9267767,0.0732233,0.82664074,
                                                       0.17335926,0.96193977,0.03806023,
                                                       0.85355339,0.14644661],'d'))


        squeezed_probs = np.zeros( mevt_squeeze.num_final_elements(), 'd')
        mgateset.bulk_fill_probs(squeezed_probs, mevt_squeeze)
        print("squeezed probs = ",squeezed_probs)
        print("lookup = ",mlookup)
        self.assertArraysAlmostEqual(probs, squeezed_probs)

        split_probs = np.zeros( mevt_split.num_final_elements(), 'd')
        mgateset.bulk_fill_probs(split_probs, mevt_split)
        print("split probs = ",split_probs)
        print("lookup = ",mlookup_splt)
        for i in range(4): #then number of original strings (num final strings)
            self.assertArraysAlmostEqual(probs[mlookup[i]], split_probs[mlookup_splt[i]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
