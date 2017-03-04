import unittest
import pygsti
import numpy as np

import pygsti
from ..testutils import BaseTestCase, compare_files, temp_files

from pygsti.construction import std1Q_XY
from pygsti.construction import std2Q_XYCNOT

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

        t = TreeClass()
        t.initialize([""] + gateLabels, strs)

        #normal order
        #print("\n".join([ "%d: %s -> %s" % (k,str(iStart),str(rem)) for k,(iStart,rem) in enumerate(t)]))

        #eval order
        evorder = t.get_evaluation_order()
        for i,k in enumerate(evorder):
            iStart,rem = t[k]
            #print("%d: %d -> %s [%s]" % (i,evorder.index(iStart) if (iStart is not None) else -1,str(rem),str(strs[k])))

        print("Number of strings = ",len(strs))
        print("Number of labels = ",sum(map(len,strs)))
        print("Number of prep fiducials = ",len(prepStrs))
        print("Number of germs = ",len(germs))
        print("Number of germs*maxLen = ",len(germs)*maxLens[-1])
        print("Number of fids*germs*maxLen = ",len(prepStrs)*len(germs)*maxLens[-1])

        if(TreeClass == pygsti.obj.MapEvalTree):
            ops = 0
            for iStart, remainder in t:
                ops += len(remainder)
            print("Number of apply ops = ", ops)
            self.assertEqual(ops, t.get_num_applies())
        #else specific tests for MatrixEvalTree?

        #Split using numSubTrees
        gsl1 = t.generate_gatestring_list()
        t.split(numSubTrees=5)
        gsl2 = t.generate_gatestring_list()
        self.assertEqual(gsl1,gsl2)

        unpermuted_list = t.generate_gatestring_list(permute=False)
        self.assertTrue(t.is_split())
        
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
        t2.initialize([""] +gateLabels, strs)

        gsl1 = t.generate_gatestring_list()
        t2.split(maxSubTreeSize=maxSize)
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




if __name__ == '__main__':
    unittest.main(verbosity=2)
