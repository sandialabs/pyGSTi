""" Defines the QubitGraph class and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import collections as _collections
from scipy.sparse.csgraph import floyd_warshall as _fw

class QubitGraph(object):
    """ Graph data structure TODO: docstrings for whole module!"""

    @classmethod
    def common_graph(cls, nQubits=0, geometry="line", directed=True):
        edges = []
        if nQubits >= 2: 
            if geometry in ("line","ring"):
                for i in range(nQubits-1):
                    edges.append((i,i+1))
                if nQubits > 2 and geometry == "ring":
                    edges.append((nQubits-1,0))
            elif geometry in ("grid","torus"):
                s = int(round(_np.sqrt(nQubits)))
                assert(nQubits >= 4 and s*s == nQubits), \
                    "`nQubits` must be a perfect square >= 4"
                #row links
                for irow in range(s):
                    for icol in range(s):
                        if icol+1 < s:
                            edges.append((irow*s+icol, irow*s+icol+1)) #link right
                        elif geometry == "torus" and s > 2:
                            edges.append((irow*s+icol, irow*s+0))
                            
                        if irow+1 < s:
                            edges.append((irow*s+icol, (irow+1)*s+icol)) #link down
                        elif geometry == "torus" and s > 2:
                            edges.append((irow*s+icol, 0+icol))
            else:
                raise ValueError("Invalid `geometry`: %s" % geometry)
        return cls(list(range(nQubits)), initial_edges=edges, directed=directed)
        
    
    def __init__(self, qubit_labels, initial_connectivity=None, initial_edges=None, directed=True):
        self.nqubits = len(qubit_labels)
        self.directed = directed

        # Mapping: node labels -> connectivity matrix index (fixed from here forward)
        self._nodeinds = _collections.OrderedDict([(lbl,i) for i,lbl in enumerate(qubit_labels)])

        # Connectivity matrix (could be sparse in future)
        self._connectivity = _np.zeros((self.nqubits,self.nqubits),dtype=bool)

        if initial_connectivity is not None:
            assert(initial_edges is None),"Cannot specify `initial_connectivity` and `initial_edges`!"
            assert(initial_connectivity.shape == self._connectivity.shape), \
                "`initial_connectivity must have shape %s" % str(self._connectivity.shape)
            self._connectivity[:,:] = initial_connectivity

        if initial_edges is not None:
            assert(initial_connectivity is None),"Cannot specify `initial_connectivity` and `initial_edges`!"
            self.add_edges(initial_edges)
            
        self._dirty = True #because we haven't computed paths yet (no need)
        self._distance_matrix = None
        self._predecessors = None
                
    def _refresh_dists_and_predecessors(self):
        if self._dirty:
            self._distance_matrix, _self.predecessors = _fw(
                self._connectivity,return_predecessors=True, 
                directed=self.directed, unweighted=False) # TIM - why use unweighted=False?

    def __getitem__(self, key):
        node1,node2 = key
        return self.is_directly_connected(node1,node2)
        
    def __setitem__(self, key, val):
        node1,node2 = key
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        if (not self.directed) and i > j:
            self._connectivity[j,i] = bool(val)
        else:
            self._connectivity[i,j] = bool(val)
        self.dirty = True

    def get_node_names(self):
        """ 
        Returns a tuple of node labels (correponding to integer indices where
        appropriate, e.g. for :method:`shortest_path_distance_matrix`)
        """
        return tuple(self._nodeinds.keys())
        
    def add_edges(self, edges):
        """ Add connections (list of tuple pairs) to graph """
        for node1, node2 in edges:
            self.add(node1, node2, directed)

    def add_edge(self, node1, node2):
        """ Add connection between node1 and node2 """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        assert(i != j), "Cannot add an edge from a node to itself!"
        if not self.directed and i > j: # undirected => only fill upper triangle (i < j)
            i,j = j,i 
        self._connectivity[i,j] = True
        self._dirty = True
        
    def edges(self):
        ret = set()
        for ilbl,i in self._nodeinds.items():
            for jlbl,j in self._nodeinds.items():
                if self._connectivity[i,j]: 
                    ret.add( (ilbl,jlbl) ) # i < j when undirected 
        return sorted(list(ret))
    
    def radius(self, base_nodes, max_hops):
        """ 
        Returns a (sorted) array of node labels that can be reached
        from traversing at most `max_hops` edges starting
        from a node (vertex) in base_nodes.
        """
        ret = set()
        assert(max_hops >= 0)
        
        def traverse(start, hops_left):
            ret.add(start)
            if hops_left <= 0: return
            i = self._nodeinds[start]
            for jlbl,j in self._nodeinds.items():
                if self._indices_connected(i,j):
                    traverse(jlbl, hops_left-1)
                
        for node in base_nodes:
            traverse(node,max_hops)
        return sorted(list(ret))

    def connected_combos(self, possible_nodes, size):
        count = 0
        for selected_nodes in _itertools.combinations(possible_nodes, size):
            if self.are_glob_connected(selected_nodes): count += 1
        return count

    def _indices_connected(self,i,j):
        if self.directed or i <= j:
            return bool(self._connectivity[i,j])
        else: # graph is NOT directed and i > j, so check for j->i link
            return bool(self._connectivity[j,i])

    def is_connected(self, node1, node2):
        """ Is node1 connected to node2 (does there exist a path between them) """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._predecessors[i,j] >= 0
        
    def is_directly_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        return self._indices_connected(i,j)

    def are_glob_connected(self, nodes):
        """
        Are all the nodes in `indices` connected to at least
        one other node in `indices`?
        """
        if len(nodes) < 2: return True # 0 or 1 nodes are "connected"        

        for node in nodes: #check
            if node not in self._nodeinds: return False

        glob = set()
        def add_to_glob(node):
            glob.add(node)
            i = self._nodeinds[node]
            for jlbl,j in self._nodeinds.items():
                if self._indices_connected(i,j) and jlbl not in glob:
                    add_to_glob(jlbl)
        
        add_to_glob(nodes[0])
        return bool(glob == set(nodes))

    def shortest_path(self, node1, node2):
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        node_labels = tuple(self._nodeinds.keys()) # relies on fact that
          # _nodeinds is an OrderedDict with keys ordered by nodelabel index

        # Find the shortest path between node1 and node2
        # (following the chain in self._predecessors until we arrive at node1)
        shortestpath = [node2]
        current_node = node2
        current_index = j
                
        while current_index != i:            
            preceeding_index = self._predecessors[i,current_index]
            assert(preceeding_index >= 0), \
                "Nodes %s and %s are not connected - no shortest path." % (str(node1),str(node2))
              #NOTE: above assert is unnecessary - testing is_connected(node1,node2) initially is fine.
            shortestpath.insert(0,node_labels[preceeding_index])
            current_index = preceeding_index            
        return shortestpath


    def shortest_path_edges(self, node1, node2):
        """ Like shortest_path, but returns a list of (nodeA,nodeB) tuples,
            where the first tuple's nodeA == `node1` and the final tuple's
            nodeB == `node2`.
        """
        path = self.shortest_path(node1,node2)
        return [ (path[i],path[i+1]) for i in range(len(path)-1)]

    def shortest_path_intersect(self, node1, node2, nodesToIntersect):
        """ Returns whether the shortest path between `node1` and `node2`
            contains any of the nodes in `nodesToIntersect`
        """
        path_set = set(self.shortest_path(node1,node2))
        return len(path_set.intersection(nodesToIntersect)) > 0
    
    def shortest_path_distance(self, node1, node2):
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._distance_matrix[i,j]

    def shortest_path_distance_matrix(self):
        """ 
        Returns a distance of shortest path distances, indexed by the
        integer-index of each node label (as specified to __init__).
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._distance_matrix.copy()

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, str(self._connectivity))
