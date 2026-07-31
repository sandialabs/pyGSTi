"""Circuit "snippers" for QPANNs.

A snipper is a locality/feature-selection specification used by QPANN rate-prediction layers.
For each elementary error generator, it returns a list of indices into the circuit encoder's
per-layer feature vector that should be used as inputs when predicting that generator's rate.
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
    hops: int | None = None, *, input_is: str = 'auto', adjacency_matrix: _np.ndarray | None = None,
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
        raw graph Laplacian or adjacency matrix (`numpy.ndarray`, nested list/tuple, or
        `scipy.sparse` matrix; see `input_is`). If the graph object carries its own qubit
        labels (a labeled `networkx`/`igraph`/`graph_tool` graph, a `QubitGraph`, or a
        `QubitProcessorSpec`), those labels must agree with `encoder.pspec.qubit_labels`
        (order does not matter in that case); a bare matrix's rows/columns are always
        positional and are matched to `encoder.pspec.qubit_labels` by position. See
        `pygsti.extras.ml.graphtools.qubit_graph_to_networkx` for the full list of accepted
        types and exactly how they're interpreted.

    hops : int
        The number of steps on the qubit graph to take.

    input_is : {'auto', 'laplacian', 'adjacency'}, optional
        Only consulted when `qubit_graph` is a bare matrix; see
        `pygsti.extras.ml.graphtools.qubit_graph_to_networkx`.

    adjacency_matrix : numpy.ndarray, optional
        Deprecated alias for `qubit_graph`. Specify only one of the two.

    Returns 
    -------
    list[list[int]]
        A list of lists, of the same length as `error_generators`. The ith element of this list
        is the indices in the layer encoding used by `encoder` that a QPANN should look at for
        predicting the rate of the corresponding error generator. This list is in the correct
        format to be passed to an initialization of a QPANN, as the `snipper` argument.
        
    Notes
    -----
    "Within `hops` steps" is determined by true (unweighted) shortest-path graph distance
    (computed via breadth-first search). Earlier versions of this function instead computed a
    graph Laplacian `L = D - A` and used `L**hops` to infer which nodes are within `hops` steps
    (via nonzero entries); that was a heuristic (and, for an isolated qubit with no edges, an
    inaccurate one -- it dropped the qubit's own index from its own list for any `hops >= 1`).
    """
    qubit_graph = _graphtools._resolve_qubit_graph_arg(qubit_graph, adjacency_matrix, 'adjacency_matrix')
    if hops is None:
        raise TypeError("Missing required argument: 'hops'.")

    from pygsti.processors import QubitProcessorSpec
    assert isinstance(encoder.pspec, QubitProcessorSpec)
    qubit_labels = list(encoder.pspec.qubit_labels)

    # For each qubit (identified by its position in qubit_labels, matching Pauli-string
    # position), find the positions of all qubits within `hops` hops of it on `qubit_graph`
    # (always including the qubit itself, regardless of hops or its degree).
    nodes_within_hops = _graphtools.qubits_within_hops(
        qubit_graph, hops, qubit_labels=qubit_labels, include_self=True, input_is=input_is)
    #
    # Init the list that this function will return, specifying the relevant encoding indices for each error generator in `error_generators`
    encoding_indices = []
    for error_generator in error_generators:
        # The Pauli(s) that label the error gen, as strings containing 'I', 'X', 'Y', and 'Z'.
        # This is a 1-tuple for 'H'/'S' error generators, or a 2-tuple of two DISTINCT Paulis
        # for 'C'/'A' error generators (see "A Taxonomy of Small Errors", Sec. V.C-V.D). In
        # either case, we take the UNION of the qubits acted on by every Pauli in the tuple,
        # since (per the same paper, Sec. VIII) the "support"/"weight" of a 'C'/'A' generator
        # C_{P,Q}/A_{P,Q} is defined as the union of P's and Q's individual qubit supports.
        pauli_strings = error_generator[1]
        # The following commented-out line is *wrong* but it used to be in the code, so leaving it here 
        # but commented out for now. It is unclear if somehow this was the correct thing to do in older
        # versions of the QPANN code before my (Tim's) rewrite.
        # pauli_string = pauli_string[::-1] # for reverse indexing
        #
        # The indices of each `pauli` that are not equal to 'I' are the qubits that PART of the
        # error acts on; take the union across all Pauli(s) in this error generator's tuple.
        qubits_acted_on_by_error = sorted(set().union(*[
            set(_np.where(_np.array(list(pauli_string)) != 'I')[0]) for pauli_string in pauli_strings
        ]))
        # All the qubits that are within `hops` steps on the graph of the qubits acted on by the error
        relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
        # The encoding indices that encode what is happening to these qubits
        relevant_encoding_indices = encoder.indices_for_qubits(list(relevant_qubits))
        # Add to the list specifying the relevant encoding indices for each error generator in `error_generators`
        encoding_indices.append(relevant_encoding_indices)

    return encoding_indices
