"""
Defines the QubitGraph class and supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import itertools as _itertools
import collections as _collections
from scipy.sparse.csgraph import floyd_warshall as _fw


class QubitGraph(object):
    """
    A directed or undirected graph data structure used to represent geometrical layouts of qubits or qubit gates.

    Qubits are nodes in the graph (and can be labeled), and edges represent the
    ability to perform one or more types of gates between qubits (equivalent,
    usually, to geometrical proximity).

    Parameters
    ----------
    qubit_labels : list
        A list of string or integer labels of the qubits.  The length of
        this list equals the number of qubits (nodes) in the graph.

    initial_connectivity : numpy.ndarray, optional
        A (nqubits, nqubits) boolean or integer array giving the initial
        connectivity of the graph.  If an integer array, then 0 indicates
        no edge and positive integers indicate present edges in the
        "direction" given by the positive integer.  For example `1` may
        corresond to "left" and `2` to "right".  Names must be associated
        with these directions using `direction_names`.  If a boolean array,
        if there's an edge from qubit `i` to `j` then
        `initial_connectivity[i,j]=True` (integer indices of qubit
        labels are given by their position in `qubit_labels`).  When
        `directed=False`, only the upper triangle is used.

    initial_edges : list
        A list of `(qubit_label1, qubit_label2)` 2-tuples or
        `(qubit_label1, qubit_label2, direction)` 3-tuples
        specifying which edges are initially present.  `direction`
        can either be a positive integer, similar to those used in
        `initial_connectivity` (in which case `direction_names` must
        be specified) or a string labeling the direction, e.g. `"left"`.

    directed : bool, optional
        Whether the graph is directed or undirected.  Directions can only
        be used when `directed=True`.

    direction_names : iterable, optional
        A list (or tuple, etc) of string-valued direction names such as
        `"left"` or `"right"`.  These strings label the directions
        referenced by index in either `initial_connectivity` or
        `initial_edges`, and this argument is required whenever such
        indices are used.
    """

    @classmethod
    def common_graph(cls, num_qubits=0, geometry="line", directed=True, qubit_labels=None, all_directions=False):
        """
        Create a QubitGraph that is one of several standard types of graphs.

        Parameters
        ----------
        num_qubits : int, optional
            The number of qubits (nodes in the graph).

        geometry : {"line","ring","grid","torus"}
            The type of graph.  What these correspond to
            should be self-evident.

        directed : bool, optional
            Whether the graph is directed or undirected.

        qubit_labels : iterable, optional
            The labels for the qubits.  Must be of length `num_qubits`.
            If None, then the integers from 0 to `num_qubits-1` are used.

        all_directions : bool, optional
            Whether to include edges with all directions.  Typically it
            only makes sense to set this to `True` when `directed=True` also.

        Returns
        -------
        QubitGraph
        """
        qls = tuple(range(num_qubits)) if (qubit_labels is None) else qubit_labels
        assert(len(qls) == num_qubits), "Invalid `qubit_labels` arg - length %d! (expected %d)" % (len(qls), num_qubits)
        edges = []
        if num_qubits >= 2:
            if geometry in ("line", "ring"):
                for i in range(num_qubits - 1):
                    edges.append((qls[i], qls[i + 1], "right") if directed else (qls[i], qls[i + 1]))
                    if all_directions:
                        edges.append((qls[i + 1], qls[i], "left") if directed else (qls[i + 1], qls[i]))
                if num_qubits > 2 and geometry == "ring":
                    edges.append((qls[num_qubits - 1], qls[0], "right") if directed else (qls[num_qubits - 1], qls[0]))
                    if all_directions:
                        edges.append((qls[0], qls[num_qubits - 1], "left")
                                     if directed else (qls[0], qls[num_qubits - 1]))
            elif geometry in ("grid", "torus"):
                s = int(round(_np.sqrt(num_qubits)))
                assert(num_qubits >= 4 and s * s == num_qubits), \
                    "`num_qubits` must be a perfect square >= 4"
                #row links
                for irow in range(s):
                    for icol in range(s):
                        if icol + 1 < s:
                            q0, q1 = qls[irow * s + icol], qls[irow * s + icol + 1]
                            edges.append((q0, q1, "right") if directed else (q0, q1))  # link right
                            if all_directions:
                                edges.append((q1, q0, "left") if directed else (q1, q0))  # link left
                        elif geometry == "torus" and s > 2:
                            q0, q1 = qls[irow * s + icol], qls[irow * s + 0]
                            edges.append((q0, q1, "right") if directed else (q0, q1))
                            if all_directions:
                                edges.append((q1, q0, "left") if directed else (q1, q0))

                        if irow + 1 < s:
                            q0, q1 = qls[irow * s + icol], qls[(irow + 1) * s + icol]
                            edges.append((q0, q1, "down") if directed else (q0, q1))  # link down
                            if all_directions:
                                edges.append((q1, q0, "up") if directed else (q1, q0))  # link up
                        elif geometry == "torus" and s > 2:
                            q0, q1 = qls[irow * s + icol], qls[0 + icol]
                            edges.append((q0, q1, "down") if directed else (q0, q1))
                            if all_directions:
                                edges.append((q1, q0, "up") if directed else (q1, q0))
            else:
                raise ValueError("Invalid `geometry`: %s" % geometry)
        return cls(qls, initial_edges=edges, directed=directed)

    def __init__(self, qubit_labels, initial_connectivity=None, initial_edges=None,
                 directed=True, direction_names=None):
        """
        Initialize a new QubitGraph.

        Can specify at most one of `initial_connectivity` and `initial_edges`.

        Parameters
        ----------
        qubit_labels : list
            A list of string or integer labels of the qubits.  The length of
            this list equals the number of qubits (nodes) in the graph.

        initial_connectivity : numpy.ndarray, optional
            A (nqubits, nqubits) boolean or integer array giving the initial
            connectivity of the graph.  If an integer array, then 0 indicates
            no edge and positive integers indicate present edges in the
            "direction" given by the positive integer.  For example `1` may
            corresond to "left" and `2` to "right".  Names must be associated
            with these directions using `direction_names`.  If a boolean array,
            if there's an edge from qubit `i` to `j` then
            `initial_connectivity[i,j]=True` (integer indices of qubit
            labels are given by their position in `qubit_labels`).  When
            `directed=False`, only the upper triangle is used.

        initial_edges : list
            A list of `(qubit_label1, qubit_label2)` 2-tuples or
            `(qubit_label1, qubit_label2, direction)` 3-tuples
            specifying which edges are initially present.  `direction`
            can either be a positive integer, similar to those used in
            `initial_connectivity` (in which case `direction_names` must
            be specified) or a string labeling the direction, e.g. `"left"`.

        directed : bool, optional
            Whether the graph is directed or undirected.  Directions can only
            be used when `directed=True`.

        direction_names : iterable, optional
            A list (or tuple, etc) of string-valued direction names such as
            `"left"` or `"right"`.  These strings label the directions
            referenced by index in either `initial_connectivity` or
            `initial_edges`, and this argument is required whenever such
            indices are used.
        """
        self.nqubits = len(qubit_labels)
        self.directed = directed

        #Determine whether we'll be using directions or not: set self.directions
        if initial_connectivity is not None:
            if initial_connectivity.dtype == _np.bool:
                assert(direction_names is None), \
                    "`initial_connectivity` must have *integer* indices when `direction_names` is non-None"
            else:
                #TODO: fix numpy integer-type test here
                assert(initial_connectivity.dtype == _np.int), \
                    ("`initial_connectivity` can only have dtype == bool or "
                     "int (but has dtype=%s)") % str(initial_connectivity.dtype)
                assert(direction_names is not None), \
                    "must supply `direction_names` when `initial_connectivity` contains *integers*!"
            self.directions = list(direction_names) if direction_names is not None else None
            # either a list of direction names or None, indicating no directions

        elif initial_edges is not None:
            lens = list(map(len, initial_edges))
            if len(lens) == 0:
                # set direction names if we're given any
                self.directions = list(direction_names) if direction_names is not None else None
            else:
                assert(all([x == lens[0] for x in lens])), \
                    "All elements of `initial_edges` must be tuples of *either* length 2 or 3.  You can't mix them."
                if lens[0] == 2:
                    assert(direction_names is None), \
                        "`initial_edges` elements must be 3-tuples when `direction_names` is non-None"
                elif lens[0] == 3:
                    direction_names_chk = set()
                    contains_direction_indices = False
                    for edge in initial_edges:
                        if isinstance(edge[2], int):
                            contains_direction_indices = True
                        else:
                            direction_names_chk.add(edge[2])
                    if contains_direction_indices:
                        assert(direction_names is not None), \
                            "must supply `direction_names` when `initial_edges` contains direction indices!"
                    if direction_names is not None:
                        assert(direction_names_chk.issubset(direction_names)), \
                            "Missing one or more direction names from `direction_names`!"
                    else:  # direction_name is None, and that's ok b/c no direction indices were used
                        direction_names = list(sorted(direction_names_chk))
                self.directions = direction_names  # set direction names if we're given any

        assert(self.directions is None or self.directed), "QubitGraph directions can only be used with `directed==True`"

        # effectively maps node index -> node name
        self._nodes = tuple(qubit_labels)

        # Mapping: node labels -> connectivity matrix index (fixed from here forward)
        self._nodeinds = _collections.OrderedDict([(lbl, i) for i, lbl in enumerate(qubit_labels)])

        # Connectivity matrix (could be sparse in future)
        typ = bool if self.directions is None else int
        self._connectivity = _np.zeros((self.nqubits, self.nqubits), dtype=typ)

        if initial_connectivity is not None:
            assert(initial_edges is None), "Cannot specify `initial_connectivity` and `initial_edges`!"
            assert(initial_connectivity.shape == self._connectivity.shape), \
                "`initial_connectivity must have shape %s" % str(self._connectivity.shape)
            self._connectivity[:, :] = initial_connectivity

        if initial_edges is not None:
            assert(initial_connectivity is None), "Cannot specify `initial_connectivity` and `initial_edges`!"
            self.add_edges(initial_edges)

        self._dirty = True  # because we haven't computed paths yet (no need)
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
                          directed=self.directed, direction_names=self.directions)

    def _refresh_dists_and_predecessors(self):
        if self._dirty:
            self._distance_matrix, self._predecessors = _fw(
                self._connectivity, return_predecessors=True,
                directed=self.directed, unweighted=False)  # TIM - why use unweighted=False?

    def __getitem__(self, key):
        node1, node2 = key
        return self.is_directly_connected(node1, node2)

    def __setitem__(self, key, val):
        node1, node2 = key
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        if (not self.directed) and i > j:  # undirected => no directions
            self._connectivity[j, i] = bool(val)
        elif self.directions is None:
            self._connectivity[i, j] = bool(val)
        else:  # directions are being used, so connectivity matrix contains ints
            dir_index = val if isinstance(val, int) else self.directions.index(val)
            self._connectivity[i, j] = dir_index
        self._dirty = True

    def __len__(self):
        return len(self._nodeinds)

    @property
    def node_names(self):
        """
        All the node labels of this graph.

        These correpond to integer indices where appropriate,
        e.g. for :method:`shortest_path_distance_matrix`.

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
        for edge_tuple in edges:  # edge_tuple is either (node1, node2) or (node1, node2, direction)
            self.add_edge(*edge_tuple)

    def add_edge(self, node1, node2, direction=None):
        """
        Add an edge between the qubits labeled by `node1` and `node2`.

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        direction : str or int, optional
            Either a direction name or a direction indicex

        Returns
        -------
        None
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        assert(i != j), "Cannot add an edge from a node to itself!"
        assert(self.directed or direction is None), "`direction` can only be specified on directed QuitGraphs"
        assert(bool(self.directions is None) == bool(direction is None)), "Direction existence mismatch!"
        if not self.directed and i > j:  # undirected => only fill upper triangle (i < j)
            i, j = j, i
        if self.directions is not None:
            dir_index = direction if isinstance(direction, int) else self.directions.index(direction)
            self._connectivity[i, j] = dir_index + 1  # b/c 0 means "no edge"
        else:
            self._connectivity[i, j] = True
        self._dirty = True

    def remove_edge(self, node1, node2):
        """
        Add an edge between the qubits labeled by `node1` and `node2`.

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        None
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        if not self.directed and i > j:  # undirected => only fill upper triangle (i < j)
            i, j = j, i
        assert(self._connectivity[i, j]), "Edge %s->%s doesn't exist!" % (str(node1), str(node2))
        self._connectivity[i, j] = False if (self.directions is None) else 0
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
        for ilbl, i in self._nodeinds.items():
            for jlbl, j in self._nodeinds.items():
                if self._connectivity[i, j]:
                    ret.add((ilbl, jlbl))  # i < j when undirected
                    if (not self.directed) and double_for_undirected:
                        ret.add((jlbl, ilbl))
        return sorted(list(ret))

    def radius(self, base_nodes, max_hops):
        """
        Find all the nodes reachable in `max_hops` from any node in `base_nodes`.

        Get a (sorted) array of node labels that can be reached
        from traversing at most `max_hops` edges starting
        from a node (vertex) in `base_nodes`.

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
            for jlbl, j in self._nodeinds.items():
                if self._indices_connected(i, j):
                    traverse(jlbl, hops_left - 1)

        for node in base_nodes:
            traverse(node, max_hops)
        return sorted(list(ret))

    def connected_combos(self, possible_nodes, size):
        """
        Computes the number of different connected subsets of `possible_nodes` containing `size` nodes.

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

    def _indices_connected(self, i, j):
        """ Whether nodes *indexed* by i and j are directly connected """
        if self.directed or i <= j:
            return bool(self._connectivity[i, j])
        else:  # graph is NOT directed and i > j, so check for j->i link
            return bool(self._connectivity[j, i])

    def is_connected(self, node1, node2):
        """
        Is `node1` connected to `node2` (does there exist a path of any length between them?)

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        bool
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._predecessors[i, j] >= 0

    def has_edge(self, edge):
        """
        Is `edge` an edge in this graph.

        Note that if this graph is undirected, either node
        order in `edge` will return True.

        Parameters
        ----------
        edge : tuple
            (node1,node2) tuple specifying the edge.

        Returns
        -------
        bool
        """
        return self.is_directly_connected(edge[0], edge[1])

    def is_directly_connected(self, node1, node2):
        """
        Is `node1` *directly* connected to `node2` (does there exist an edge  between them?)

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        bool
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        return self._indices_connected(i, j)

    def are_glob_connected(self, nodes):
        """
        Does there exist a path from every node in `nodes` to every other node in `nodes`?

        That is, do these nodes form a connected set?

        Parameters
        ----------
        nodes : list
            A list of node (qubit) labels.

        Returns
        -------
        bool
        """
        if len(nodes) < 2: return True  # 0 or 1 nodes are "connected"

        for node in nodes:  # check
            if node not in self._nodeinds: return False

        def add_to_glob(glob, node):
            glob.add(node)
            i = self._nodeinds[node]
            for jlbl, j in self._nodeinds.items():
                if self._indices_connected(i, j) and jlbl not in glob:
                    add_to_glob(glob, jlbl)

        if not self.directed:
            # then just check that we can get from nodes[0] to all the others:
            glob = set(); add_to_glob(glob, nodes[0])
            return set(nodes).issubset(glob)
        else:
            # we need to check that, starting at *any* initial node, we can
            # reach all the others:
            for node in nodes:
                glob = set(); add_to_glob(glob, node)
                if not set(nodes).issubset(glob): return False
            return True

    def _brute_get_all_connected_sets(self, n):
        """
        Computes all connected sets of `n` qubits using a brute-force approach.

        Note that for a large device with this will be often be an
        unreasonably large number of sets of qubits, and so
        the run-time of this method will be unreasonable.

        Parameters
        ----------
        n : int
            The number of qubits within each set.

        Returns
        -------
        list
            All sets of `n` connected qubits.
        """
        connectedqubits = []
        for combo in _itertools.combinations(self.node_names, n):
            if self.subgraph(list(combo)).are_glob_connected(combo):
                connectedqubits.append(combo)

        return connectedqubits

    def find_all_connected_sets(self):
        """
        Finds all subgraphs (connected sets of vertices) up to the full graph size.

        Graph edges are treated as undirected.

        Returns
        -------
        dict
            A dictionary with integer keys.  The value of key `k` is a
            list of all the subgraphs of length `k`.  A subgraph is given
            as a tuple of sorted vertex labels.
        """
        def add_neighbors(neighbor_dict, max_subgraph_size, subgraph_vertices, visited_vertices, output_set):
            """ x holds subgraph so far.  y holds vertices already processed. """
            if len(subgraph_vertices) == max_subgraph_size: return output_set  # can't add any more vertices; exit now.

            T = set()  # vertices to process - those connected to vertices in x
            if len(subgraph_vertices) == 0:  # special starting case
                T.update(neighbor_dict.keys())  # all vertices are connected to the "nothing"/empty set of vertices.
            else:  # normal case
                for v in subgraph_vertices:
                    T.update(filter(lambda w: (w not in visited_vertices and w not in subgraph_vertices),
                                    neighbor_dict[v]))  # add neighboring vertices we haven't already processed.
            V = set(visited_vertices)
            for v in T:
                subgraph_vertices.add(v)
                output_set.add(frozenset(subgraph_vertices))
                add_neighbors(neighbor_dict, max_subgraph_size, subgraph_vertices, V, output_set)
                subgraph_vertices.remove(v)
                V.add(v)

        def addedge(a, b, neighbor_dict):
            neighbor_dict[a].append(b)
            neighbor_dict[b].append(a)

        def group_subgraphs(subgraph_list):
            processed_subgraph_dict = _collections.defaultdict(list)
            for subgraph in subgraph_list:
                k = len(subgraph)
                subgraph_as_list = list(subgraph)
                subgraph_as_list.sort()
                subgraph_as_tuple = tuple(subgraph_as_list)
                processed_subgraph_dict[k].append(subgraph_as_tuple)
            return processed_subgraph_dict

        neighbor_dict = _collections.defaultdict(list)
        directed_edge_list = self.edges()
        undirected_edge_list = list(set([frozenset(edge) for edge in directed_edge_list]))
        undirected_edge_list = [list(edge) for edge in undirected_edge_list]

        for edge in undirected_edge_list:
            addedge(edge[0], edge[1], neighbor_dict)
        k_max = len(self)  # number of vertices in this graph
        output_set = set()
        add_neighbors(neighbor_dict, k_max, set(), set(), output_set)
        grouped_subgraphs = group_subgraphs(output_set)
        return grouped_subgraphs

    def shortest_path(self, node1, node2):
        """
        Get the shortest path between two nodes (qubits).

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        list
            A list of the node labels to traverse.
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        node_labels = tuple(self._nodeinds.keys())  # relies on fact that
        # _nodeinds is an OrderedDict with keys ordered by nodelabel index

        # Find the shortest path between node1 and node2
        # (following the chain in self._predecessors until we arrive at node1)
        shortestpath = [node2]
        current_index = j

        while current_index != i:
            preceeding_index = self._predecessors[i, current_index]
            assert(preceeding_index >= 0), \
                "Nodes %s and %s are not connected - no shortest path." % (str(node1), str(node2))
            #NOTE: above assert is unnecessary - testing is_connected(node1,node2) initially is fine.
            shortestpath.insert(0, node_labels[preceeding_index])
            current_index = preceeding_index
        return shortestpath

    def shortest_path_edges(self, node1, node2):
        """
        Like :method:`shortest_path`, but returns a list of (nodeA,nodeB) tuples.

        These tuples define a path from `node1` to `node2`, so the first tuple's
        nodeA == `node1` and the final tuple's nodeB == `node2`.

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        list
            A list of the edges (2-tuples of node labels) to traverse.
        """
        path = self.shortest_path(node1, node2)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def shortest_path_intersect(self, node1, node2, nodes_to_intersect):
        """
        Check whether the shortest path between `node1` and `node2` contains any of the nodes in `nodes_to_intersect`.

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        nodes_to_intersect : list
            A list of node labels.

        Returns
        -------
        bool
            True if the shortest path intersects any node in `nodeToIntersect`.
        """
        path_set = set(self.shortest_path(node1, node2))
        return len(path_set.intersection(nodes_to_intersect)) > 0

    def shortest_path_distance(self, node1, node2):
        """
        Get the distance of the shortest path between `node1` and `node2`.

        Parameters
        ----------
        node1 : str or int
            Qubit (node) label.

        node2 : str or int
            Qubit (node) label.

        Returns
        -------
        int
        """
        i, j = self._nodeinds[node1], self._nodeinds[node2]
        self._refresh_dists_and_predecessors()
        return self._distance_matrix[i, j]

    def shortest_path_distance_matrix(self):
        """
        Returns a matrix of shortest path distances.

        This matrix is indexed by the integer-index of each node label (as
        specified to __init__).  The list of index-ordered node labels is given
        by :method:`node_names`.

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
        Returns a matrix of predecessors used to construct the shortest path between two nodes.

        This matrix is indexed by the integer-index of each node label (as
        specified to __init__).  The list of index-ordered node labels is given
        by :method:`node_names`.

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
        Return a graph that includes only `nodes_to_keep` and the edges between them.

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
            labelmap = {old: i for i, old in enumerate(nodes_to_keep)}
        else:
            qubit_labels = nodes_to_keep

        edges = []
        for edge in self.edges():
            if edge[0] in nodes_to_keep and edge[1] in nodes_to_keep:
                if reset_nodes:
                    edges.append((labelmap[edge[0]], labelmap[edge[1]]))
                else:
                    edges.append(edge)

        return QubitGraph(qubit_labels, initial_edges=edges, directed=self.directed)

    def resolve_relative_nodelabel(self, relative_nodelabel, target_labels):
        """
        Resolve a "relative nodelabel" into an actual node in this graph.

        Relative node labels can use "@" to index elements of `target_labels`
        and can contain "+<dir>" directives to move along directions defined
        in this graph.

        Parameters
        ----------
        relative_nodelabel : int or str
            An absolute or relative node-label.  For example:
            `0`, `"@0"`, `"@0+right"`, `"@1+left+up"`

        target_labels : list or tuple
            A list of (absolute) node labels present in this graph that may
            be referred to using the "@" syntax within `relative_nodelabel`.

        Returns
        -------
        int or str
        """
        if relative_nodelabel in self.node_names:
            return relative_nodelabel  # relative_nodelabel is a valid absolute node label
        elif isinstance(relative_nodelabel, str) and relative_nodelabel.startswith("@"):
            # @<target_index> or @<target_index>+<direction>
            parts = relative_nodelabel.split('+')
            target_index = int(parts[0][1:])  # we know parts[0] starts with @ and rest should be an int index
            start_node = target_labels[target_index]
            return self.move_in_directions(start_node, parts[1:])  # parts[1:] are (optional) directions
        else:
            raise ValueError("Unknown node: %s" % str(relative_nodelabel))

    def move_in_directions(self, start_node, directions):
        """
        The node you end up on after moving in `directions` from `start_node`.

        Parameters
        ----------
        start_node : str or int
            Qubit (node) label.

        directions : iterable
            A sequence of direction names.

        Returns
        -------
        str or int or None
            The ending node label or `None` if the directions were invalid.
        """
        node = start_node
        for direction in directions:
            node = self.move_in_direction(node, direction)
            if node is None:
                return None
        return node

    def move_in_direction(self, start_node, direction):
        """
        Get the node that is one step in `direction` of `start_node`.

        Parameters
        ----------
        start_node : int or str
            the starting point (a node label of this graph)

        direction : str or int
            the name of a direction or its index within this graphs
            `.directions` member.

        Returns
        -------
        str or int or None
            the node in the given direction or `None` if there is no
            node in that direction (e.g. if you reach the end of a
            chain).
        """
        assert(self.directions is not None), "This QubitGraph doesn't have directions!"
        i = self._nodeinds[start_node]
        dir_index = direction if isinstance(direction, int) else self.directions.index(direction)
        for j, d in enumerate(self._connectivity[i, :]):
            if d == dir_index:
                return self._nodes[j]
        return None  # No node in this direction

    def __str__(self):
        dirstr = "Directed" if self.directed else "Undirected"
        s = dirstr + ' Qubit Graph w/%d qubits.  Nodes = %s\n' % (self.nqubits, str(self._nodeinds))
        s += ' Edges = ' + str(self.edges()) + '\n'
        return s
