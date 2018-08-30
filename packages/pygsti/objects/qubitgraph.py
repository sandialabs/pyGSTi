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
    """
    A directed or undirected graph data structure used to represent geometrical
    layouts of qubits or qubit gates.

    Qubits are nodes in the graph (and can be labeled), and edges represent the
    ability to perform one or more types of gates between qubits (equivalent,
    usually, to geometrical proximity).
    """

    @classmethod
    def common_graph(cls, nQubits=0, geometry="line", directed=True):
        """
        Create a QubitGraph that is one of several standard types of graphs.

        Parameters
        ----------
        nQubits : int, optional
            The number of qubits (nodes in the graph).

        geometry : {"line","ring","grid","torus"}
            The type of graph.  What these correspond to
            should be self-evident.

        directed : bool, optional
            Whether the graph is directed or undirected.

        Returns
        -------
        QubitGraph
        """
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
        """
        Initialize a new QubitGraph.

        Can specify at most one of `initial_connectivity` and `initial_edges`.

        Parameters
        ----------
        qubit_labels : list
            A list of string or integer labels of the qubits.  The length of
            this list equals the number of qubits (nodes) in the graph.

        initial_connectivity : numpy.ndarray, optional
            A (nqubits, nqubits) boolean array giving the initial connectivity
            of the graph.  That is, if there's an edge from qubit `i` to `j`,
            then `initial_connectivity[i,j]=True` (integer indices of qubit
            labels are given by their position in `qubit_labels`).  When
            `directed=False`, only the upper triangle is used.  

        initial_edges : list
            A list of `(qubit_label1, qubit_label2)` 2-tuples specifying which
            edges are initially present.

        directed : bool, optional
            Whether the graph is directed or undirected.
        """
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

    def copy(self):
        """
        Make a copy of this graph.

        Returns
        -------
        QubitGraph
        """
        return QubitGraph(list(self._nodeinds.keys()),
                          initial_connectivity=self._connectivity,
                          directed=self.directed)
        
                
    def _refresh_dists_and_predecessors(self):
        if self._dirty:
            self._distance_matrix, self._predecessors = _fw(
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
        Returns a tuple of node labels (correponding to integer indices
        where appropriate, e.g. for :method:`shortest_path_distance_matrix`)

        Returns
        -------
        tuple
        """
        return tuple(self._nodeinds.keys())
        
    def add_edges(self, edges):
        """ 
        Add edges (list of tuple pairs) to graph.

        Parameters
        ----------
        edges : list
            A list of `(qubit_label1, qubit_label2)` 2-tuples.

        Returns
        -------
        None
        """
        for node1, node2 in edges:
            self.add_edge(node1, node2)

    def add_edge(self, node1, node2):
        """ 
        Add an edge between the qubits labeled by `node1` and `node2`.

        Parameters
        ----------
        node1,node2 : object
            Qubit (node) labels - typically strings or integers.

        Returns
        -------
        None
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        assert(i != j), "Cannot add an edge from a node to itself!"
        if not self.directed and i > j: # undirected => only fill upper triangle (i < j)
            i,j = j,i 
        self._connectivity[i,j] = True
        self._dirty = True

    def remove_edge(self, node1, node2):
        """ 
        Add an edge between the qubits labeled by `node1` and `node2`.

        Parameters
        ----------
        node1,node2 : object
            Qubit (node) labels - typically strings or integers.

        Returns
        -------
        None
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        if not self.directed and i > j: # undirected => only fill upper triangle (i < j)
            i,j = j,i
        assert(self._connectivity[i,j]), "Edge %s->%s doesn't exist!" % (str(node1),str(node2))
        self._connectivity[i,j] = False
        self._dirty = True

        
    def edges(self, double_for_undirected=False):
        """
        Get a list of the edges in this graph as 2-tuples of node/qubit labels).
        
        When undirected, the index of the 2-tuple's first label will always be
        less than its second unless `double_for_undirected == True`, in which 
        case both directed edges are included.  The edges are sorted (by label
        *index*) in ascending order.

        Parameters
        ----------
        double_for_undirected : bool, optional
            Whether, for the case of an undirected graph, two 2-tuples, giving
            both edge directions, should be included in the returned list.

        Returns
        -------
        list
        """
        ret = set()
        for ilbl,i in self._nodeinds.items():
            for jlbl,j in self._nodeinds.items():
                if self._connectivity[i,j]:
                    ret.add( (ilbl,jlbl) ) # i < j when undirected
                    if (not self.directed) and double_for_undirected:
                        ret.add( (jlbl,ilbl) )
        return sorted(list(ret))
    
    def radius(self, base_nodes, max_hops):
        """ 
        Get a (sorted) array of node labels that can be reached
        from traversing at most `max_hops` edges starting
        from a node (vertex) in base_nodes.

        Parameters
        ----------
        base_nodes : iterable
            A list of node/qubit labels giving the possible starting locations.

        max_hops : int
            The maximum number of hops (see above).

        Returns
        -------
        list
            A list of the node labels reachable from `base_nodes` in at most
            `max_hops` edge traversals.
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
        """
        Computes the number of different connected subsets of `possible_nodes`
        containing `size` nodes.

        Parameters
        ----------
        possible_nodes : list
            A list of node (qubit) labels.

        size : int
            The size of the connected subsets being sought (counted).

        Returns
        -------
        int
        """
        count = 0
        for selected_nodes in _itertools.combinations(possible_nodes, size):
            if self.are_glob_connected(selected_nodes): count += 1
        return count

    def _indices_connected(self,i,j):
        """ Whether nodes *indexed* by i and j are directly connected """
        if self.directed or i <= j:
            return bool(self._connectivity[i,j])
        else: # graph is NOT directed and i > j, so check for j->i link
            return bool(self._connectivity[j,i])

    def is_connected(self, node1, node2):
        """ 
        Is `node1` connected to `node2` (does there exist a path of any length
        between them?)

        Parameters
        ----------
        node1, node2 : object
            The node (qubit) labels to check.
        
        Returns
        -------
        bool
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._predecessors[i,j] >= 0


    def has_edge(self, edge):
        """
        Is `edge` an edge in this graph.  Note that if this graph is
        undirected, either node order in `edge` will return True.

        Parameters
        ----------
        edge : tuple
            (node1,node2) tuple specifying the edge.

        Returns
        -------
        bool
        """
        return self.is_directly_connected(edge[0],edge[1])
    
    
    def is_directly_connected(self, node1, node2):
        """ 
        Is `node1` *directly* connected to `node2` (does there exist an edge
        between them?)

        Parameters
        ----------
        node1, node2 : object
            The node (qubit) labels to check.
        
        Returns
        -------
        bool
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        return self._indices_connected(i,j)

    def are_glob_connected(self, nodes):
        """
        Does there exist a path from every node in `nodes`
        to every other node in `nodes`.  That is, do these
        nodes form a connected set?

        Parameters
        ----------
        nodes : list
            A list of node (qubit) labels.
        
        Returns
        -------
        bool
        """
        if len(nodes) < 2: return True # 0 or 1 nodes are "connected"        

        for node in nodes: #check
            if node not in self._nodeinds: return False

        def add_to_glob(glob,node):
            glob.add(node)
            i = self._nodeinds[node]
            for jlbl,j in self._nodeinds.items():
                if self._indices_connected(i,j) and jlbl not in glob:
                    add_to_glob(glob,jlbl)
            
        if not self.directed:
            # then just check that we can get from nodes[0] to all the others:
            glob = set(); add_to_glob(glob,nodes[0])
            return bool(glob == set(nodes))
        else:
            # we need to check that, starting at *any* initial node, we can
            # reach all the others:
            for node in nodes:
                glob = set(); add_to_glob(glob,node)
                if not bool(glob == set(nodes)): return False
            return True
            

    def shortest_path(self, node1, node2):
        """
        Get the shortest path between two nodes (qubits).

        Parameters
        ----------
        node1, node2 : object
            Node (qubit) labels, usually integers or strings.
        
        Returns
        -------
        list
            A list of the node labels to traverse.
        """
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
        """ 
        Like :method:`shortest_path`, but returns a list of (nodeA,nodeB)
        tuples, where the first tuple's nodeA == `node1` and the final tuple's
        nodeB == `node2`.

        Parameters
        ----------
        node1, node2 : object
            Node (qubit) labels, usually integers or strings.
        
        Returns
        -------
        list
            A list of the edges (2-tuples of node labels) to traverse.
        """
        path = self.shortest_path(node1,node2)
        return [ (path[i],path[i+1]) for i in range(len(path)-1)]

    def shortest_path_intersect(self, node1, node2, nodesToIntersect):
        """ 
        Determine  whether the shortest path between `node1` and `node2`
        contains any of the nodes in `nodesToIntersect`.

        Parameters
        ----------
        node1, node2 : object
            Node (qubit) labels, usually integers or strings.

        nodesToIntersect : list
            A list of node labels.
        
        Returns
        -------
        bool
            True if the shortest path intersects any node in `nodeToIntersect`.
        """
        path_set = set(self.shortest_path(node1,node2))
        return len(path_set.intersection(nodesToIntersect)) > 0
    
    def shortest_path_distance(self, node1, node2):
        """
        Get the distance of the shortest path between `node1` and `node2`.

        Parameters
        ----------
        node1, node2 : object
            Node (qubit) labels, usually integers or strings.
        
        Returns
        -------
        int
        """
        i,j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._distance_matrix[i,j]

    def shortest_path_distance_matrix(self):
        """ 
        Returns a matrix of shortest path distances, indexed by the
        integer-index of each node label (as specified to __init__).  The list
        of index-ordered node labels is given by :method:`get_node_names`.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape (n,n) where n is the number of nodes in
            this graph.
        """
        self._refresh_dists_and_predecessors()
        return self._distance_matrix.copy()

    def shortest_path_predecessor_matrix(self):
        """ 
        Returns a matrix of predecessors used to construct the
        shortest path between two nodes, indexed by the
        integer-index of each node label (as specified to __init__).  The list
        of index-ordered node labels is given by :method:`get_node_names`.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape (n,n) where n is the number of nodes in
            this graph.
        """
        self._refresh_dists_and_predecessors()
        return self._predecessors.copy()


    def subgraph(self, nodes_to_keep, reset_nodes=False):
        """
        Return a graph that includes only `nodes_to_keep` and 
        the edges between them. 

        Parameters
        ----------
        nodes_to_keep : list
            A list of node labels defining the subgraph to return.

        reset_nodes : bool, optional
            If True, nodes of returned subgraph are relabelled to
            be the integers starting at 0 (in 1-1 correspondence
            with the ordering in `nodes_to_keep`).

        Returns
        -------
        QubitGraph
        """
        if reset_nodes:
            qubit_labels = list(range(len(nodes_to_keep)))
            labelmap = { old: i for i,old in enumerate(nodes_to_keep) }
        else:
            qubit_labels = nodes_to_keep

        edges = []
        for edge in self.edges():
            if edge[0] in nodes_to_keep and edge[1] in nodes_to_keep:
                if reset_nodes:
                    edges.append( (labelmap[edge[0]],labelmap[edge[1]]) )
                else:
                    edges.append( edge )

        return QubitGraph(qubit_labels, initial_edges=edges, directed=self.directed)

    def __str__(self):
        dirstr = "Directed" if self.directed else "Undirected"
        s = dirstr + ' Qubit Graph w/%d qubits.  Nodes = %s\n' % (self.nqubits,str(self._nodeinds))
        s += ' Edges = ' + str(self.edges()) + '\n'
        return s
