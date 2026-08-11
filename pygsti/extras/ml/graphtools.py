"""Graph-library-agnostic qubit connectivity utilities for `pygsti.extras.ml`.

This module lets the rest of `pygsti.extras.ml` (`errgentools`, `snippers`) accept a qubit
connectivity graph in essentially any common representation:

  * a `networkx.Graph`/`DiGraph`/`MultiGraph`/`MultiDiGraph`,
  * an `igraph.Graph` (optional third-party dependency; only inspected via duck-typing, never
    imported by this module),
  * a `graph_tool.Graph` (same: optional, duck-typed, never imported here),
  * a `pygsti.baseobjs.QubitGraph`,
  * a `pygsti.processors.QubitProcessorSpec` (its 2-qubit-gate connectivity is used, via
    `compute_2Q_connectivity()`),
  * a raw adjacency matrix, as a `numpy.ndarray`, nested list/tuple, or `scipy.sparse` matrix, or
  * an explicit edge list (via `qubit_graph_from_edges`).

A bare matrix must be a plain adjacency matrix (non-negative entries); a graph Laplacian is not
accepted (see `_matrix_to_adjacency`'s error message if you have one -- it costs nothing to
convert a Laplacian `L = D - A` to an adjacency matrix `A` yourself, whereas supporting both
representations ambiguously here would cost real complexity for no real benefit).

Since `networkx` is already a hard pyGSTi dependency (see `pyproject.toml`), it is used as the
single canonical internal representation: every accepted input is first coerced to a plain
undirected `networkx.Graph` (via `qubit_graph_to_networkx`), and every other function in this
module is built on top of that one coercion step. `igraph`/`graph-tool` objects are recognized
by inspecting the input object itself (its class's module name and a couple of expected
methods) -- this module never does `import igraph`/`import graph_tool` anywhere, so those
packages remain purely optional, user-installed conveniences and are never required.

Qubits are always identified by *position* (matching the convention used throughout
`errgentools`/`snippers`/`encoding`, where Pauli-string position `i` <-> qubit `i`). When a
graph object has its own meaningful node labels (a labeled `networkx.Graph`, a `QubitGraph`, a
`QubitProcessorSpec`), those labels define the position ordering (position `i` <-> the `i`-th
label in the object's natural node order) unless an explicit `qubit_labels` ordering is
supplied. A bare matrix or edge list has no labels of its own, so its rows/columns/positions are
always `0..n-1` unless `qubit_labels` is supplied to attach labels positionally.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from typing import Any

import numpy as _np
import networkx as _nx

from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph

__all__ = [
    'qubit_graph_to_networkx',
    'qubit_graph_from_edges',
    'qubit_graph_adjacency_matrix',
    'within_hops_matrix',
    'qubits_within_hops',
]

# Absolute tolerance used when checking a bare matrix for (disallowed) negative entries, and
# when treating small numerical residues as exact zeros.
_ATOL = 1e-8


# ----------------------------------------------------------------------------------------------
# Type detection / extraction helpers. None of these import igraph or graph_tool: third-party
# graph objects are recognized purely by inspecting `type(obj).__module__` and a couple of
# expected attributes on the object the caller already constructed (and therefore already has
# the corresponding package installed for).
# ----------------------------------------------------------------------------------------------

def _module_root(obj: Any) -> str:
    """The top-level package name of `obj`'s class, e.g. `'igraph'`, `'networkx'`, `'builtins'`."""
    return type(obj).__module__.split('.')[0]


def _is_processor_spec(obj: Any) -> bool:
    """True iff `obj` is a `pygsti.processors.QubitProcessorSpec`."""
    try:
        from pygsti.processors import QubitProcessorSpec as _QPS
    except ImportError:  # pragma: no cover - pygsti.processors is always importable in practice
        return False
    return isinstance(obj, _QPS)


def _extract_networkx(g: _nx.Graph) -> tuple[list, list[tuple]]:
    """Node list (native/insertion order) and edge list from any networkx graph type."""
    nodes = list(g.nodes())
    edges = [(u, v) for u, v in g.edges() if u != v]
    return nodes, edges


def _extract_qubit_graph(g: _QubitGraph) -> tuple[list, list[tuple]]:
    """Node list and edge list from a `pygsti.baseobjs.QubitGraph` (directed or undirected)."""
    nodes = list(g.node_names)
    # `include_directions=False` is correct regardless of `g.directed`: we only care about
    # *whether* two qubits are connected, not the direction, and every edge returned here
    # becomes a single undirected edge below -- so a directed QubitGraph with only a (u, v)
    # (and not (v, u)) edge still ends up correctly connecting u and v.
    edges = list(g.edges(include_directions=False))
    return nodes, edges


def _extract_igraph(g: Any) -> tuple[list, list[tuple]]:
    """Node list and edge list from an `igraph.Graph` (detected via duck-typing)."""
    n = g.vcount()
    if 'name' in g.vertex_attributes():
        nodes = list(g.vs['name'])
    else:
        nodes = list(range(n))
    edges = [(nodes[i], nodes[j]) for i, j in g.get_edgelist()]
    return nodes, edges


def _extract_graph_tool(g: Any) -> tuple[list, list[tuple]]:
    """Node list and edge list from a `graph_tool.Graph` (detected via duck-typing)."""
    n = g.num_vertices()
    nodes = None
    vertex_properties = getattr(g, 'vertex_properties', {})
    for key in ('name', 'label'):
        if key in vertex_properties:
            prop = vertex_properties[key]
            nodes = [prop[v] for v in g.vertices()]
            break
    if nodes is None:
        nodes = list(range(n))
    edges = [(nodes[int(i)], nodes[int(j)]) for i, j in g.get_edges()]
    return nodes, edges


def _coerce_to_dense_matrix(obj: Any):
    """
    If `obj` looks like a square matrix (numpy array, nested list/tuple, or scipy.sparse
    matrix), return it as a dense float `numpy.ndarray`. Otherwise, return None.
    """
    try:
        import scipy.sparse as _sp
        if _sp.issparse(obj):
            obj = obj.toarray()
    except ImportError:  # pragma: no cover - scipy is a hard pygsti dependency
        pass

    if isinstance(obj, _np.ndarray):
        M = obj
    elif isinstance(obj, (list, tuple)):
        try:
            M = _np.asarray(obj)
        except (TypeError, ValueError):
            return None
    else:
        return None

    try:
        M = _np.asarray(M, dtype=float)
    except (TypeError, ValueError):
        return None
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return None
    return M


def _matrix_to_adjacency(M: _np.ndarray, atol: float = _ATOL) -> _np.ndarray:
    """
    Validate that `M` is a plain adjacency matrix and return a zero-diagonal copy of it.

    Parameters
    ----------
    M : numpy.ndarray
        A square matrix.
    atol : float, optional
        Absolute tolerance used when checking for negative entries.

    Returns
    -------
    numpy.ndarray
        A copy of `M` with its diagonal zeroed.

    Raises
    ------
    ValueError
        If `M` is not square, or has a negative off-diagonal entry (a graph Laplacian
        `L = D - A` always has off-diagonal entries `<= 0`, so this is the tell-tale sign of
        one being passed where a plain adjacency matrix -- entries `>= 0` -- is required).
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"A qubit-graph matrix must be square; got shape {M.shape}.")

    offdiag = M[~_np.eye(M.shape[0], dtype=bool)]
    if _np.any(offdiag < -atol):
        raise ValueError(
            "A bare qubit_graph matrix must be a plain adjacency matrix (off-diagonal entries "
            ">= 0), but this one has negative off-diagonal entries -- it looks like a graph "
            "Laplacian (L = D - A) instead. Convert it to an adjacency matrix first (e.g. "
            "`A = -L; numpy.fill_diagonal(A, 0)`), or pass a networkx/igraph/graph-tool graph, "
            "a pygsti QubitGraph, or a QubitProcessorSpec instead."
        )

    A = _np.array(M, copy=True)
    _np.fill_diagonal(A, 0)
    return A


def _reconcile_nodes(orig_nodes: list, edges: list[tuple], qubit_labels) -> tuple[list, list[tuple]]:
    """
    Reconcile a graph's own node list against a caller-supplied `qubit_labels` ordering.

    If `qubit_labels` is None, `orig_nodes`/`edges` are returned unchanged (the graph's own
    native node order is used as-is). Otherwise, there are two ways `orig_nodes` is allowed to
    correspond to `qubit_labels`:

      1. Direct label match: every node in `orig_nodes` is itself one of the `qubit_labels`
         (this is the common case for a labeled `networkx.Graph`, a `QubitGraph`, or a
         `QubitProcessorSpec`). Any `qubit_labels` entries absent from `orig_nodes` are added as
         isolated nodes (e.g. a qubit with no 2-qubit gates acting on it).
      2. Positional match: `orig_nodes` is exactly the integers `0..len(qubit_labels)-1` (as
         produced by a bare matrix, an unlabeled `networkx`/`igraph`/`graph_tool` graph, etc.),
         in which case node `i` is identified with `qubit_labels[i]`.

    Parameters
    ----------
    orig_nodes : list
        The node identifiers as extracted from the input graph, in that graph's native order.
    edges : list[tuple]
        The edges as extracted from the input graph (using the same node identifiers as
        `orig_nodes`).
    qubit_labels : list or None
        The desired qubit ordering/labels, or None to use `orig_nodes` as-is.

    Returns
    -------
    nodes, edges : list, list[tuple]
        `nodes` is exactly `list(qubit_labels)` (if it was not None) or `list(orig_nodes)`
        (if it was); `edges` is relabeled accordingly.
    """
    if qubit_labels is None:
        return list(orig_nodes), list(edges)

    qubit_labels = list(qubit_labels)
    orig_list = list(orig_nodes)
    orig_set = set(orig_list)
    label_set = set(qubit_labels)

    if orig_set <= label_set:
        return qubit_labels, list(edges)

    if len(orig_list) == len(qubit_labels) and orig_set == set(range(len(orig_list))):
        remap = {i: qubit_labels[i] for i in orig_list}
        return qubit_labels, [(remap[u], remap[v]) for u, v in edges]

    extra = sorted((str(x) for x in (orig_set - label_set)))
    raise ValueError(
        f"Cannot reconcile the input graph's nodes with qubit_labels={qubit_labels}: node(s) "
        f"{extra} are not present in qubit_labels, and the graph's {len(orig_list)} node(s) are "
        f"not the positional integers 0..{len(orig_list) - 1} either (which would have been "
        "interpreted as positional indices into qubit_labels)."
    )


def _unsupported_type_error(obj: Any) -> TypeError:
    return TypeError(
        f"Unsupported qubit_graph type: {type(obj)!r}. Supported types are: a networkx "
        "Graph/DiGraph/MultiGraph/MultiDiGraph, an igraph.Graph, a graph_tool.Graph, a "
        "pygsti.baseobjs.QubitGraph, a pygsti.processors.QubitProcessorSpec, a square adjacency "
        "matrix (as a numpy.ndarray, nested list/tuple, or scipy.sparse matrix), or a networkx "
        "graph produced by qubit_graph_from_edges. Note: a bare edge list is ambiguous with a "
        "matrix and is not accepted here -- use "
        "`graphtools.qubit_graph_from_edges(edges, qubit_labels)` to build a graph from one."
    )


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------

def qubit_graph_to_networkx(qubit_graph: Any, qubit_labels: list | None = None) -> _nx.Graph:
    """
    Coerce a qubit connectivity graph, in essentially any common representation, to a plain
    undirected `networkx.Graph`. This is the single coercion step every other function in this
    module is built on.

    Parameters
    ----------
    qubit_graph : graph-like
        The qubit connectivity graph. Supported types:

          * a `networkx.Graph`, `DiGraph`, `MultiGraph`, or `MultiDiGraph` (directed/multi
            graphs are converted to a plain undirected graph; edge directions and multiplicity
            are discarded, since only "is qubit i connected to qubit j" matters here),
          * an `igraph.Graph` (optional dependency; recognized without importing `igraph`
            unless you have already created one yourself). Vertex names are taken from a
            `'name'` vertex attribute if present, else from vertex index (`0..n-1`),
          * a `graph_tool.Graph` (optional dependency; likewise recognized without importing
            `graph_tool`). Vertex names are taken from a `'name'` or `'label'` vertex property
            if present, else from vertex index,
          * a `pygsti.baseobjs.QubitGraph`,
          * a `pygsti.processors.QubitProcessorSpec` (its two-qubit-gate connectivity, i.e.
            `pspec.compute_2Q_connectivity()`, is used -- *not* `pspec.qubit_graph`, which has
            no edges unless an explicit `geometry` was given when the spec was constructed),
          * a square adjacency matrix, as a `numpy.ndarray`, a nested list/tuple, or a
            `scipy.sparse` matrix (off-diagonal entries must be `>= 0`; see `_matrix_to_adjacency`
            for the error raised if a graph Laplacian is passed instead), or
          * a `networkx.Graph` already built by `qubit_graph_from_edges`.

        A bare edge list (`list[tuple]`) is deliberately *not* accepted here, since e.g.
        `[(0, 1), (1, 0)]` is ambiguous with a 2x2 matrix; use `qubit_graph_from_edges` instead.

    qubit_labels : list, optional
        The desired qubit ordering. If `qubit_graph` carries its own node labels (a labeled
        `networkx`/`igraph`/`graph_tool` graph, a `QubitGraph`, or a `QubitProcessorSpec`), those
        labels must either match `qubit_labels` exactly (as a set; any of `qubit_labels` missing
        from the graph are added as isolated nodes) or be exactly the positional integers
        `0..len(qubit_labels)-1` (in which case node `i` is identified with `qubit_labels[i]`).
        A bare matrix's rows/columns are always positional, so this second case is how
        `qubit_labels` is applied to a matrix input. If None, `qubit_graph`'s own native node
        order is used (or plain positions `0..n-1` for a bare matrix).

    Returns
    -------
    networkx.Graph
        A plain, simple, undirected graph with no self-loops. `list(G.nodes())` equals
        `list(qubit_labels)` if it was given, else `qubit_graph`'s own native node order.

    Raises
    ------
    TypeError
        If `qubit_graph` is not one of the supported types.
    ValueError
        If a matrix input is not square, has a negative off-diagonal entry (see
        `_matrix_to_adjacency`), or if `qubit_labels` cannot be reconciled with `qubit_graph`'s
        own nodes.
    """
    orig_nodes: list | None = None
    edges: list[tuple] = []

    if isinstance(qubit_graph, _nx.Graph):
        orig_nodes, edges = _extract_networkx(qubit_graph)
    elif isinstance(qubit_graph, _QubitGraph):
        orig_nodes, edges = _extract_qubit_graph(qubit_graph)
    elif _is_processor_spec(qubit_graph):
        orig_nodes, edges = _extract_qubit_graph(qubit_graph.compute_2Q_connectivity())
    elif (_module_root(qubit_graph) == 'igraph'
          and hasattr(qubit_graph, 'vcount') and hasattr(qubit_graph, 'get_edgelist')):
        orig_nodes, edges = _extract_igraph(qubit_graph)
    elif (_module_root(qubit_graph) == 'graph_tool'
          and hasattr(qubit_graph, 'num_vertices') and hasattr(qubit_graph, 'get_edges')):
        orig_nodes, edges = _extract_graph_tool(qubit_graph)

    if orig_nodes is None:
        M = _coerce_to_dense_matrix(qubit_graph)
        if M is None:
            raise _unsupported_type_error(qubit_graph)
        if qubit_labels is not None and M.shape[0] != len(qubit_labels):
            raise ValueError(
                f"qubit_labels has length {len(qubit_labels)}, but the given matrix has shape "
                f"{M.shape}; a bare matrix's rows/columns are positional, so these must match."
            )
        A = _matrix_to_adjacency(M)
        n = A.shape[0]
        orig_nodes = list(range(n))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] or A[j, i]]

    nodes, edges = _reconcile_nodes(orig_nodes, edges, qubit_labels)

    G = _nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def qubit_graph_from_edges(edges: list[tuple], qubit_labels: list) -> _nx.Graph:
    """
    Build an undirected qubit connectivity graph directly from an explicit edge list. This is
    the natural entry point when you have gate-availability-style `(control, target)` tuples
    (e.g. `pspec.availability['Gcphase']`) rather than an existing graph object.

    Parameters
    ----------
    edges : list[tuple]
        A list of `(u, v)` 2-tuples, where `u` and `v` are each elements of `qubit_labels`.

    qubit_labels : list
        The complete set of node/qubit labels, defining both the graph's node set (including
        any qubits with no edges) and the row/column order used by, e.g.,
        `qubit_graph_adjacency_matrix`.

    Returns
    -------
    networkx.Graph
        A plain, simple, undirected graph with node set exactly `qubit_labels`.
    """
    if qubit_labels is None:
        raise TypeError(
            "qubit_labels is required: it defines the full node set (including any isolated "
            "qubits) and the row/column order used elsewhere."
        )
    qubit_labels = list(qubit_labels)
    label_set = set(qubit_labels)

    G = _nx.Graph()
    G.add_nodes_from(qubit_labels)
    for edge in edges:
        try:
            u, v = edge
        except (TypeError, ValueError):
            raise ValueError(f"Each edge must be a 2-tuple (u, v); got {edge!r}.")
        if u not in label_set or v not in label_set:
            raise ValueError(f"Edge {edge!r} references a node not in qubit_labels={qubit_labels}.")
        if u != v:
            G.add_edge(u, v)
    return G


def qubit_graph_adjacency_matrix(qubit_graph: Any, qubit_labels: list | None = None) -> _np.ndarray:
    """
    Get the adjacency matrix of a qubit connectivity graph, in any of the representations
    accepted by `qubit_graph_to_networkx`.

    Parameters
    ----------
    qubit_graph : graph-like
        See `qubit_graph_to_networkx`.
    qubit_labels : list, optional
        See `qubit_graph_to_networkx`. Also fixes the row/column order of the returned matrix.

    Returns
    -------
    numpy.ndarray
        Integer adjacency matrix of shape `(n, n)`, symmetric, with entries in `{0, 1}` and a
        zero diagonal. Row/column `i` corresponds to `qubit_labels[i]` if given, else to the
        `i`-th qubit in `qubit_graph`'s own native node order.
    """
    G = qubit_graph_to_networkx(qubit_graph, qubit_labels=qubit_labels)
    nodes = list(qubit_labels) if qubit_labels is not None else list(G.nodes())
    return _nx.to_numpy_array(G, nodelist=nodes, dtype=int)


def within_hops_matrix(qubit_graph: Any, hops: int, qubit_labels: list | None = None) -> _np.ndarray:
    """
    Compute the boolean "within `hops` hops" adjacency matrix for a qubit connectivity graph:
    `close[i, j]` is True iff qubits `i` and `j` (`i != j`) are connected by a path of at most
    `hops` edges. This uses true (unweighted) shortest-path graph distance, computed via
    breadth-first search.

    Parameters
    ----------
    qubit_graph : graph-like
        See `qubit_graph_to_networkx`.
    hops : int
        Maximum hop (graph-edge) distance; must be a non-negative integer.
    qubit_labels : list, optional
        See `qubit_graph_to_networkx`. Also fixes the row/column order of the returned matrix.

    Returns
    -------
    numpy.ndarray
        Boolean `(n, n)` matrix with a False diagonal (a qubit is never considered "close to
        itself" by this function; see `qubits_within_hops` for a version that includes self).
    """
    if not isinstance(hops, (int, _np.integer)) or hops < 0:
        raise ValueError(f"hops must be a non-negative integer; got {hops!r}.")

    G = qubit_graph_to_networkx(qubit_graph, qubit_labels=qubit_labels)
    nodes = list(qubit_labels) if qubit_labels is not None else list(G.nodes())
    n = len(nodes)
    index_of = {node: i for i, node in enumerate(nodes)}

    close = _np.zeros((n, n), dtype=bool)
    for node in nodes:
        i = index_of[node]
        lengths = _nx.single_source_shortest_path_length(G, node, cutoff=int(hops))
        for other, distance in lengths.items():
            if distance > 0:
                close[i, index_of[other]] = True
    return close


def qubits_within_hops(qubit_graph: Any, hops: int, qubit_labels: list | None = None,
                        include_self: bool = True) -> list[list[int]]:
    """
    For every qubit, find the (positional indices of the) qubits within `hops` hops of it on
    the qubit connectivity graph `qubit_graph`.

    Parameters
    ----------
    qubit_graph : graph-like
        See `qubit_graph_to_networkx`.
    hops : int
        Maximum hop (graph-edge) distance; must be a non-negative integer.
    qubit_labels : list, optional
        See `qubit_graph_to_networkx`. Also fixes the position ordering used in the returned
        indices (position `i` <-> `qubit_labels[i]`).
    include_self : bool, optional
        Whether qubit `i` should always be included in its own list (regardless of `hops`).
        Default True, matching the convention used elsewhere in this package that an error
        generator's own support qubits are always "relevant to themselves".

    Returns
    -------
    list[list[int]]
        A list of length `n` (the number of qubits); the `i`-th element is a sorted list of the
        positional indices of the qubits within `hops` hops of qubit `i` (including `i` itself
        iff `include_self` is True).
    """
    close = within_hops_matrix(qubit_graph, hops, qubit_labels=qubit_labels)
    n = close.shape[0]
    result = []
    for i in range(n):
        indices = set(_np.flatnonzero(close[i, :]).tolist())
        if include_self:
            indices.add(i)
        result.append(sorted(indices))
    return result
