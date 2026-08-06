"""Circuit "snippers" for QPANNs.

A snipper is a locality/feature-selection specification used by QPANN rate-prediction layers.
For each elementary error generator, it returns a list of indices into the circuit encoder's
per-layer feature vector that should be used as inputs when predicting that generator's rate.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from typing import Any, TYPE_CHECKING

from pygsti.extras.ml import graphtools as _graphtools

if TYPE_CHECKING:
    from pygsti.extras.ml.encoding import StandardCircuitEncoder


def undirected_adjacency_matrix_from_edges(edges: list[tuple], qubit_labels: list) -> _np.ndarray:
    """
    Constructs the undirected adjacency matrix for the graph with nodes given by `qubit_labels` and edges given by `edges.

    See Also
    --------
    pygsti.extras.ml.graphtools.qubit_graph_from_edges :
        Builds a `networkx.Graph` (rather than a bare matrix) from an edge list; the more
        general entry point if you want to combine this with other graph-library objects.

    Parameters
    ----------
    edges : list list[tuple]
        List of tuples of edges `(u, v)`, where each element of each edge is an element of `qubit_labels`. (i.e. `u` and `v` are elements of `qubit_labels`)

    qubit_labels : list
        Node labels defining the ordering of rows/columns in the returned matrix.

    Returns
    -------
    numpy.ndarray
        Integer adjacency matrix of shape `(len(qubit_labels), len(qubit_labels))` with
        symmetric entries in {0,1}.
    """
    qubit_labels = list(qubit_labels)
    graph = _graphtools.qubit_graph_from_edges(edges, qubit_labels)
    return _graphtools.qubit_graph_adjacency_matrix(graph, qubit_labels=qubit_labels)


def layer_snipper_from_qubit_graph(
    error_generators: list[tuple], encoder: "StandardCircuitEncoder", qubit_graph: Any = None,
    hops: int | None = None,
) -> list[list[int]]:
    """
    Creates a "snipper" for a QPANN. This snipper will specify that, when predicting the
    error rate of an error generator G that acts non-trivially on the qubit set Q, the
    QPANN shoud look at what is occuring on all the qubits within Q and all those qubits
    within 'hops' steps of that qubit on the graph given by 'qubit_graph'. This graph can be
    the connectivity of the qubits (the qubit pairs for which there are two-qubit gates), but
    it could also specify some other kind of coupling.

    Parameters
    ----------
    error_generators : list
        A list of elementary error generators, in the same format as used by QPANNs. Each element of this
            list is a tuple. The first element of the tuple is a string specifying the error
            generator type: 'H', 'S', 'C', or 'A' (Hamiltonian, Stochastic-Pauli, Stochastic
            Pauli-Correlation, and Active, respectively; see "A Taxonomy of Small Errors",
            Blume-Kohout et al.). The second element of the tuple is a tuple of Pauli string(s)
            indexing the error: a single-element tuple for 'H'/'S' (e.g., for 4 qubits, this
            could be `('XYZI',)`), or a two-element tuple of two DISTINCT Paulis for 'C'/'A'
            (e.g. `('XIII', 'IIIY')`). `Q` (the qubit set this error generator "acts non-
            trivially on", referenced above) is the *union* of the qubits acted on by every
            Pauli in this tuple.

    encoder : CircuitEncoder
        The CircuitEncoder whose encoding this snipper will reference. Typically this will be
        an instance of a StandardCircuitEncoder, as defined in ml.encoding.py

    qubit_graph : graph-like
        The qubit connectivity graph. Accepts a `networkx.Graph`/`DiGraph`/`MultiGraph`, an
        `igraph.Graph`, a `graph_tool.Graph`, a `pygsti.baseobjs.QubitGraph`, a
        `pygsti.processors.QubitProcessorSpec` (its 2-qubit-gate connectivity is used), or a
        raw adjacency matrix (`numpy.ndarray`, nested list/tuple, or `scipy.sparse` matrix). If
        the graph object carries its own qubit labels (a labeled `networkx`/`igraph`/`graph_tool`
        graph, a `QubitGraph`, or a `QubitProcessorSpec`), those labels must agree with
        `encoder.pspec.qubit_labels` (order does not matter in that case); a bare matrix's
        rows/columns are always positional and are matched to `encoder.pspec.qubit_labels` by
        position. See `pygsti.extras.ml.graphtools.qubit_graph_to_networkx` for the full list of
        accepted types and exactly how they're interpreted.

    hops : int
        The number of steps on the qubit graph to take.

    Returns
    -------
    list[list[int]]
        A list of lists, of the same length as `error_generators`. The ith element of this list
        is the indices in the layer encoding used by `encoder` that a QPANN should look at for
        predicting the rate of the corresponding error generator. This list is in the correct
        format to be passed to an initialization of a QPANN, as the `snipper` argument.

    Notes
    -----
    "Within `hops` steps" is determined by true (unweighted) shortest-path graph distance,
    computed via breadth-first search.
    """
    if qubit_graph is None:
        raise TypeError("Missing required argument: 'qubit_graph'.")
    if hops is None:
        raise TypeError("Missing required argument: 'hops'.")

    from pygsti.processors import QubitProcessorSpec
    assert isinstance(encoder.pspec, QubitProcessorSpec)
    qubit_labels = list(encoder.pspec.qubit_labels)

    # For each qubit (identified by its position in qubit_labels, matching Pauli-string
    # position), find the positions of all qubits within `hops` hops of it on `qubit_graph`
    # (always including the qubit itself, regardless of hops or its degree).
    nodes_within_hops = _graphtools.qubits_within_hops(
        qubit_graph, hops, qubit_labels=qubit_labels, include_self=True)

    # For each error generator, find the relevant encoding indices.
    encoding_indices = []
    for error_generator in error_generators:
        # The Pauli(s) that label the error gen: a 1-tuple for 'H'/'S', or a 2-tuple of two
        # DISTINCT Paulis for 'C'/'A' (see "A Taxonomy of Small Errors", Sec. V.C-V.D). We take
        # the UNION of the qubits acted on by every Pauli in the tuple, since (per the same
        # paper, Sec. VIII) that union is the defined support/weight of a 'C'/'A' generator.
        # Pauli-string position i always corresponds to qubit i (no reverse-indexing here).
        pauli_strings = error_generator[1]
        qubits_acted_on_by_error = sorted(set().union(*[
            set(_np.where(_np.array(list(pauli_string)) != 'I')[0]) for pauli_string in pauli_strings
        ]))
        # All the qubits that are within `hops` steps on the graph of the qubits acted on by the error
        relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
        # The encoding indices that encode what is happening to these qubits
        relevant_encoding_indices = encoder.indices_for_qubits(list(relevant_qubits))
        encoding_indices.append(relevant_encoding_indices)

    return encoding_indices
