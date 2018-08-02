import unittest
import pygsti
import numpy as np

import pygsti
from ..testutils import BaseTestCase, compare_files, temp_files

from pygsti.construction import std1Q_XY
from pygsti.construction import std2Q_XYCNOT
from pygsti.objects.gatemapcalc import GateMapCalc

class EvalTreeTestCase(BaseTestCase):

    def setUp(self):
        super(EvalTreeTestCase, self).setUp()

    def test_map_tree_1Q(self):
        self.helper_tree(pygsti.obj.MapEvalTree,True)

    def test_map_tree_2Q(self):
        self.helper_tree(pygsti.obj.MapEvalTree,False)

    def test_matrix_tree_1Q(self):
        self.helper_tree(pygsti.obj.MatrixEvalTree,True)

    def test_matrix_tree_2Q(self):
        self.helper_tree(pygsti.obj.MatrixEvalTree,False)

    def test_base_tree(self):
        raw_tree = pygsti.obj.EvalTree()
        with self.assertRaises(NotImplementedError):
            raw_tree.initialize(None)
        with self.assertRaises(NotImplementedError):
            raw_tree.generate_gatestring_list()
        with self.assertRaises(NotImplementedError):
            raw_tree.split(None)

    def helper_tree(self, TreeClass, b1Q):
        if b1Q:
            gs_target = std1Q_XY.gs_target
            prepStrs = std1Q_XY.fiducials
            measStrs = std1Q_XY.fiducials
            germs = std1Q_XY.germs
            maxLens = [1,4]
            #maxLens = [1,2,4,8,16,32,64,128,256,512,1024]
        else:
            gs_target = std2Q_XYCNOT.gs_target
            prepStrs = std2Q_XYCNOT.prepStrs
            measStrs = std2Q_XYCNOT.effectStrs
            germs = std2Q_XYCNOT.germs
            maxLens = [1,2,4]
    
        gateLabels = list(gs_target.gates.keys())
        strs = pygsti.construction.make_lsgst_experiment_list(
            gateLabels, prepStrs, measStrs, germs, maxLens, includeLGST=False)
        pygsti.tools.remove_duplicates_in_place(strs)
        #print("\n".join(map(str,strs)))

        compiled_gatestrings, lookup, outcome_lookup, nEls = \
                    gs_target.compile_gatestrings(strs)
        self.assertTrue(isinstance(compiled_gatestrings, dict))

        t = TreeClass()
        t.initialize(compiled_gatestrings)
        nStrs = t.num_final_strings()
        self.assertEqual(nStrs, len(compiled_gatestrings))
        self.assertEqual(t.final_slice(None),slice(0,nStrs)) #trivial since t is not split

        #normal order
        #print("\n".join([ "%d: %s -> %s" % (k,str(iStart),str(rem)) for k,(iStart,rem) in enumerate(t)]))

        #eval order
        evorder = t.get_evaluation_order()
        for i,k in enumerate(evorder):
            #print("item %d = %s" % (i,t[k]))
            assert(len(t[k]) in (2,3))
            #iStart,rem,iCache = t[k]
            #print("%d: %d -> %s [%s]" % (i,evorder.index(iStart) if (iStart is not None) else -1,str(rem),str(strs[k])))

        print("Number of strings = ",len(strs))
        print("Number of labels = ",sum(map(len,strs)))
        print("Number of prep fiducials = ",len(prepStrs))
        print("Number of germs = ",len(germs))
        print("Number of germs*maxLen = ",len(germs)*maxLens[-1])
        print("Number of fids*germs*maxLen = ",len(prepStrs)*len(germs)*maxLens[-1])

        if(TreeClass == pygsti.obj.MapEvalTree):
            ops = 0
            for iStart, remainder, iCache in t:
                ops += len(remainder)
            print("Number of apply ops = ", ops)
            self.assertEqual(ops, t.get_num_applies())

            t2 = t.copy()
            t2.squeeze(4)
            t2.squeeze(0) #special case
        else: # specific tests for MatrixEvalTree?
            t.get_min_tree_size() #just make sure it runs...

            #REMOVED - this error isn't applicable now, since an eval tree computes the
            #  distinct gate labels in the gatestrings it's given.
            ##Creation failure b/c of unknown gate labels (for now just matrix eval tree)
            #with self.assertRaises(AssertionError):
            #    tbad = TreeClass()
            #    compiled_strs,_,_,_ =gs_target.compile_gatestrings([ (), ('Gnotpresent',)])
            #    tbad.initialize(compiled_strs )

        #Split using numSubTrees
        gsl1 = t.generate_gatestring_list()
        lookup2 = t.split(lookup, numSubTrees=5)
        gsl2 = t.generate_gatestring_list()
        self.assertEqual(gsl1,gsl2)

        unpermuted_list = t.generate_gatestring_list(permute=False)
        self.assertTrue(t.is_split())

        dummy = np.random.rand(len(gsl1))
        dummy_computational = t.permute_original_to_computation(dummy)
        dummy2 = t.permute_computation_to_original(dummy_computational)
        self.assertArraysAlmostEqual(dummy,dummy2)
        
        subtrees = t.get_sub_trees()
        for i,st in enumerate(subtrees):
            #print("Subtree %d: applies = %d, size = %d" % (i,st.get_num_applies(),len(st)))
            fslc = st.final_slice(t)
            sub_gsl = st.generate_gatestring_list(permute=False) #permute=False not necessary though, since subtree is not split it's elements are not permuted
            self.assertEqual(sub_gsl,unpermuted_list[fslc])
            

        #Split using maxSubTreeSize
        maxSize = 25
        print("Splitting with max subtree size = ",maxSize)
        t2 = TreeClass()
        t2.initialize(compiled_gatestrings)

        gsl1 = t.generate_gatestring_list()
        lookup2 = t2.split(lookup, maxSubTreeSize=maxSize)
        gsl2 = t.generate_gatestring_list()
        self.assertEqual(gsl1,gsl2)

        unpermuted_list = t2.generate_gatestring_list(permute=False)

        self.assertTrue(t2.is_split())
        
        subtrees2 = t2.get_sub_trees()
        for i,st in enumerate(subtrees2):
            #print("Subtree %d: applies = %d, size = %d" % (i,st.get_num_applies(),len(st)))
            fslc = st.final_slice(t2)
            sub_gsl = st.generate_gatestring_list(permute=False) #permute=False not necessary though, since subtree is not split it's elements are not permuted
            self.assertEqual(sub_gsl,unpermuted_list[fslc])

            
        #Test invalid split arguments
        t3 = TreeClass()
        t3.initialize(compiled_gatestrings)            
        with self.assertRaises(ValueError):
            t3.split(lookup, maxSubTreeSize=10, numSubTrees=10) #can't specify both
        with self.assertRaises(ValueError):
            t3.split(lookup, numSubTrees=0) #numSubTrees must be > 0

        #Creation of a tree with duplicate strings
        if b1Q:
            strs_with_dups = [(),
                              ('Gx',),
                              ('Gy',),
                              ('Gx','Gy'),
                              ('Gx','Gy'),
                              (),
                              ('Gx','Gx','Gx'),
                              ('Gx','Gy','Gx')]
            compiled_gatestrings2, lookup2, outcome_lookup2, nEls2 = \
                        gs_target.compile_gatestrings(strs_with_dups)
            tdup = TreeClass()
            tdup.initialize(compiled_gatestrings2)
        
        



    def test_mapevaltree(self):
        # An additional specific test added from debugging mapevaltree splitting
        mgateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        mgateset._calcClass = GateMapCalc
        
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
                                   (None, ('Gx',), 0),
                                   (None, ('Gy', 'Gy'), None)])
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
                                   (None, ('Gx',), 0),
                                   (None, ('Gy', 'Gy'), None)])
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
        
        self.assertEqual(mevt_squeeze[:], [(None, ('Gx', 'Gy'), 0),
                                           (0, ('Gy',), None),
                                           (None, ('Gx',), None),
                                           (None, ('Gy', 'Gy'), None)])
        self.assertEqual(mevt_squeeze.cache_size(),maxCacheSize)
        self.assertEqual(mevt_squeeze.get_evaluation_order(),[2, 0, 1, 3])
        self.assertEqual(mevt_squeeze.num_final_strings(),4)
        
          #SPLIT
        mevt_split = mevt.copy()
        mlookup_splt = mevt_split.split(mlookup,numSubTrees=4)
        print("Split tree = ",mevt_split)
        print("new lookup = ",mlookup_splt)
        print()

        self.assertEqual(mevt_split[:], [(None, ('Gx',), 0),
                                         (0, ('Gy',), 1),
                                         (1, ('Gy',), None),
                                         (None, ('Gy', 'Gy'), None)])
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
