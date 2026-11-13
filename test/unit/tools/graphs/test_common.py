#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import unittest
from pygsti.tools.graphs import (
    order, canonical_edges, find_neighbors, max_degree,
)
from ...util import BaseCase


class CanonicalEdgesTester(BaseCase):
    """``canonical_edges`` dedup and orientation."""

    def test_canonical_edges_behavior(self):
        import pygsti.tools.graphs as graphs
        self.assertIn('canonical_edges', graphs.__all__)
        self.assertIs(graphs.canonical_edges, canonical_edges)
        
        self.assertEqual(canonical_edges([(0, 1), (1, 0), (1, 2), (2, 1)]), [(0, 1), (1, 2)])
        self.assertEqual(canonical_edges([(1, 0), (2, 1)]), [(0, 1), (1, 2)])
        
        # Orientation matches order() coloring convention
        edges = canonical_edges([(3, 1), (1, 0)])
        self.assertEqual(edges, [order(3, 1), order(1, 0)])
        
        # First encounter order is preserved, not sorted
        self.assertEqual(canonical_edges([(2, 1), (1, 0)]), [(1, 2), (0, 1)])
        self.assertEqual(canonical_edges([]), [])


class FindNeighborsTester(BaseCase):
    """``find_neighbors`` adjacency construction."""

    def test_find_neighbors_behavior(self):
        import pygsti.tools.graphs as graphs
        self.assertIn('find_neighbors', graphs.__all__)
        self.assertIs(graphs.find_neighbors, find_neighbors)

        # Each edge is recorded from both endpoints
        self.assertEqual(find_neighbors((0, 1, 2), [(0, 1), (1, 2)]), {0: [1], 1: [0, 2], 2: [1]})

        # Writing both orientations does not duplicate neighbors
        one_directional = find_neighbors((0, 1, 2), [(0, 1), (1, 2)])
        two_directional = find_neighbors((0, 1, 2), [(0, 1), (1, 0), (1, 2), (2, 1)])
        self.assertEqual(two_directional, one_directional)

        # Isolated vertices get empty neighbor lists
        self.assertEqual(find_neighbors((0, 1, 2), [(0, 1)]), {0: [1], 1: [0], 2: []})

        # Multiple neighbors accumulate in edge order
        self.assertEqual(find_neighbors((0, 1, 2, 3), [(1, 0), (1, 2), (1, 3)]),
                         {0: [1], 1: [0, 2, 3], 2: [1], 3: [1]})


class MaxDegreeTester(BaseCase):
    """``max_degree`` degree computation."""

    def test_max_degree_behavior(self):
        import pygsti.tools.graphs as graphs
        self.assertIn('max_degree', graphs.__all__)
        self.assertIs(graphs.max_degree, max_degree)

        self.assertEqual(max_degree({0: [1], 1: [0, 2], 2: [1]}), 2)
        self.assertEqual(max_degree({0: [], 1: [], 2: []}), 0)
        self.assertEqual(max_degree({}), 0)
