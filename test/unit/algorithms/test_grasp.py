import numpy as np

from ..util import BaseCase

from pygsti.algorithms import grasp


# Helper functions to parameterize grasp.do_grasp
def someScoreFn(elements_subset):
    num_t = sum([el.count('t') for el in elements_subset])
    num_els = len(elements_subset)
    return num_els**2 - num_t


def indices_of_candidates(list_of_elementsubsets):
    M = min(len(list_of_elementsubsets), 2)  # take "top" two candidates if there are two
    return list(range(M))


def getNeighbors(binary_element_vec):
    neighbor_vecs = []
    for i, b in enumerate(binary_element_vec):
        if b:
            v = binary_element_vec.copy()
            if i + 1 < len(binary_element_vec) and not v[i + 1]:
                v[i + 1] = 1
                neighbor_vecs.append(v)

            v = binary_element_vec.copy()
            if i > 0 and not v[i - 1]:
                v[i - 1] = 1
                neighbor_vecs.append(v)

            v = binary_element_vec.copy()
            if (i + 1 < len(binary_element_vec) and v[i + 1]) or \
               (i > 0 and v[i - 1]):
                v[i] = 0
                neighbor_vecs.append(v)

    return neighbor_vecs


def feasibleFn(elements_subset):
    return bool(len(elements_subset) > 2)


class GraspTester(BaseCase):
    def setUp(self):
        super(GraspTester, self).setUp()
        self.elements = ("""
Many designers find the flexboxes easier to use than boxes. Without a
lot of work, div's frequently rose to the top of a page when designers
did not want them to---so for example, sticking a footer to the bottom
of a page was difficult. The widths and heights of flexboxes vary to
adapt to the display space, holding the lower down elements in
place. Flexbox logic also asks whether you want div's to accrue to the
right or on the bottom. The display order of flexbox elements is
independent of their order in the source code
        """).split()

    def test_do_grasp(self):
        result = grasp.do_grasp(
            self.elements, greedy_score_fn=someScoreFn,
            rcl_fn=indices_of_candidates, local_score_fn=someScoreFn,
            get_neighbors_fn=getNeighbors, final_score_fn=someScoreFn,
            iterations=10, feasible_threshold=None,
            feasible_fn=feasibleFn, initial_elements=None, seed=1234,
            verbosity=3
        )
        # TODO assert correctness

    def test_do_grasp_with_initial_elements(self):
        initial_elements = np.zeros(len(self.elements))
        initial_elements[0] = initial_elements[2] = initial_elements[10] = 1.0 #some initial state

        result = grasp.do_grasp(
            self.elements, greedy_score_fn=someScoreFn,
            rcl_fn=indices_of_candidates, local_score_fn=someScoreFn,
            get_neighbors_fn=getNeighbors, final_score_fn=someScoreFn,
            iterations=10, feasible_threshold=None,
            feasible_fn=feasibleFn, initial_elements=initial_elements,
            seed=1234, verbosity=3
        )
        # TODO assert correctness

    def test_do_grasp_raises_on_initial_element_size_mismatch(self):
        with self.assertRaises(ValueError):
            grasp.do_grasp(
                self.elements, greedy_score_fn=someScoreFn,
                rcl_fn=indices_of_candidates, local_score_fn=someScoreFn,
                get_neighbors_fn=getNeighbors, final_score_fn=someScoreFn,
                iterations=10, feasible_threshold=None,
                feasible_fn=feasibleFn, initial_elements=[0], seed=1234,
                verbosity=3
            )

    def test_do_grasp_raises_on_missing_feasible_function_and_threshold(self):
        with self.assertRaises(ValueError):
            grasp.do_grasp(
                self.elements, greedy_score_fn=someScoreFn,
                rcl_fn=indices_of_candidates, local_score_fn=someScoreFn,
                get_neighbors_fn=getNeighbors, final_score_fn=someScoreFn,
                iterations=10, feasible_threshold=None,
                feasible_fn=None, initial_elements=[0], seed=1234,
                verbosity=3
            )
