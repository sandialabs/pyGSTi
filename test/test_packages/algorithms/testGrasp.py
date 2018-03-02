from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest

import numpy as np
import sys, os

from ..testutils import BaseTestCase, temp_files, compare_files

class GraspTestCase(BaseTestCase):

    def setUp(self):
        super(GraspTestCase, self).setUp()

    def test_grasp(self):

        elements = "Many designers find the flexboxes easier to use than boxes. Without a lot of work, div's frequently rose to the top of a page when designers did not want them to---so for example, sticking a footer to the bottom of a page was difficult. The widths and heights of flexboxes vary to adapt to the display space, holding the lower down elements in place. Flexbox logic also asks whether you want div's to accrue to the right or on the bottom. The display order of flexbox elements is independent of their order in the source code".split()

        def someScoreFn(elements_subset):
            num_t = sum([el.count('t') for el in elements_subset])
            num_els = len(elements_subset)
            return num_els**2 - num_t

        def indices_of_candidates(list_of_elementsubsets):
            M = min(len(list_of_elementsubsets), 2) #take "top" two candidates if there are two
            return list(range(M))

        def getNeighbors(binary_element_vec):
            neighbor_vecs = []
            for i,b in enumerate(binary_element_vec):
                if b:
                    v = binary_element_vec.copy()
                    if i+1 < len(binary_element_vec) and not v[i+1]:
                        v[i+1] = 1
                        neighbor_vecs.append(v)
                        
                    v = binary_element_vec.copy()
                    if i > 0 and not v[i-1]:
                        v[i-1] = 1
                        neighbor_vecs.append(v)
                        
                    v = binary_element_vec.copy()
                    if (i+1 < len(binary_element_vec) and v[i+1]) or \
                       (i > 0 and v[i-1]):
                        v[i] = 0
                        neighbor_vecs.append(v)
                        
            return neighbor_vecs

        def feasibleFn(elements_subset):
            return bool(len(elements_subset) > 2)

        #no initial elements => all zeros => no neighbors
        pygsti.algorithms.grasp.do_grasp(elements, greedyScoreFn=someScoreFn, rclFn=indices_of_candidates,
                                         localScoreFn=someScoreFn, getNeighborsFn=getNeighbors,
                                         finalScoreFn=someScoreFn, iterations=10, feasibleThreshold=None,
                                         feasibleFn=feasibleFn, initialElements=None, seed=1234, verbosity=3)

        initial_elements = np.zeros(len(elements))
        initial_elements[0] = initial_elements[2] = initial_elements[10] = 1.0 #some initial state
        pygsti.algorithms.grasp.do_grasp(elements, greedyScoreFn=someScoreFn, rclFn=indices_of_candidates,
                                         localScoreFn=someScoreFn, getNeighborsFn=getNeighbors,
                                         finalScoreFn=someScoreFn, iterations=10, feasibleThreshold=None,
                                         feasibleFn=feasibleFn, initialElements=initial_elements, seed=1234, verbosity=3)

        with self.assertRaises(ValueError):
            #bad initialElements (not len(elements) long)
            pygsti.algorithms.grasp.do_grasp(elements, greedyScoreFn=someScoreFn, rclFn=indices_of_candidates,
                                             localScoreFn=someScoreFn, getNeighborsFn=getNeighbors,
                                             finalScoreFn=someScoreFn, iterations=10, feasibleThreshold=None,
                                             feasibleFn=feasibleFn, initialElements=[0], seed=1234, verbosity=3)

        with self.assertRaises(ValueError):
            #need feasibleFn or feasibleThreshold
            pygsti.algorithms.grasp.do_grasp(elements, greedyScoreFn=someScoreFn, rclFn=indices_of_candidates,
                                             localScoreFn=someScoreFn, getNeighborsFn=getNeighbors,
                                             finalScoreFn=someScoreFn, iterations=10, feasibleThreshold=None,
                                             feasibleFn=None, initialElements=None, seed=1234, verbosity=3)



if __name__ == '__main__':
    unittest.main(verbosity=2)
