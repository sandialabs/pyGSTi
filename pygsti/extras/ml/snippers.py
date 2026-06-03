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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygsti.extras.ml.encoding import StandardCircuitEncoder

def undirected_adjacency_matrix_from_edges(edges: list[tuple], qubit_labels: list) -> _np.ndarray:
    """
    Constructs the undirected adjacency matrix for the graph with nodes given by `qubit_labels` and edges given by `edges.

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
    adjacency_matrix = _np.zeros((len(qubit_labels), len(qubit_labels)), int)
    for edge in edges:
        adjacency_matrix[qubit_labels.index(edge[0]), qubit_labels.index(edge[1])] = 1
        adjacency_matrix[qubit_labels.index(edge[1]), qubit_labels.index(edge[0])] = 1
    return adjacency_matrix

# FUTURE TODO: This function assumes only H and S errors. Will need to update if/when
# code is updated to add in C and A errors.
def layer_snipper_from_qubit_graph(error_generators: list[tuple], encoder: "StandardCircuitEncoder", adjacency_matrix: _np.ndarray, hops: int) -> list[list[int]]:
    """
    Creates a "snipper" for a QPANN. This snipper will specify that, when predicting the
    error rate of an error generator G that acts non-trivially on the qubit set Q, the 
    QPANN shoud look at what is occuring on all the qubits within Q and all those qubits
    within 'hops' steps of that qubit on the graph given by the 'adjacency_matrix'. This
    adjacency matrix can be the connectivity of the qubits (the qubit pairs for which there
    are two-qubit gates), but it could also be an adjacency matrix that specifies some other
    kind of coupling.

    Parameters
    ----------
    error_generators : list
        A list of elementary error generators, in the same format as used by QPANNs. Each element of this 
            list is a tuple. The first element of the tuple is a string specifying the error
            generator type: 'H' or 'S', for Hamiltonian and stochastic errors (currently active
            and Pauli-correlation errors are not supported). The second element of
            the tuple is a single-element tuple where that single element is a string for the 
            Pauli indexing the error (e.g., for 4 qubits, this could be 'XYZI').

    encoder : CircuitEncoder
        The CircuitEncoder whose encoding this snipper will reference. Typically this will be
        an instance of a StandardCircuitEncoder, as defined in ml.encoding.py

    adjacency_matrix : numpy.array
        A numpy array specifying the adjacency matrix of the qubits. 

    hops : int
        The number of steps on the adjacency graph to take

    Returns 
    -------
    list[list[int]]
        A list of lists, of the same length as `error_generators`. The ith element of this list
        is the indices in the layer encoding used by `encoder` that a QPANN should look at for
        predicting the rate of the corresponding error generator. This list is in the correct
        format to be passed to an initialization of a QPANN, as the `snipper` argument.
        
    Notes
    -----
    This function computes a graph Laplacian `L = D - A` and uses `L**hops` to infer which
    nodes are within `hops` steps (via nonzero entries). This is a heuristic; depending on
    graph structure, using powers of the adjacency matrix may be more conventional.
    """
    # Compute the set of qubits that are within `hops` steps on the adjacency graph of each qubit,
    # by computing the Lapalacian and taking its `hops` power.
    degree_matrix = _np.diag(_np.sum(adjacency_matrix, axis = 1))
    laplacian = degree_matrix - adjacency_matrix
    laplace_power = _np.linalg.matrix_power(laplacian, hops)
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(encoder.pspec, QubitProcessorSpec)
    nodes_within_hops = [list(_np.arange(encoder.pspec.num_qubits)[abs(laplace_power[i, :]) > 0]) for i in range(encoder.pspec.num_qubits)]
    #
    # Init the list that this function will return, specifying the relevant encoding indices for each error generator in `error_generators`
    encoding_indices = []
    for error_generator in error_generators:
        # The Pauli that labels the error gen, as a string containing 'I', 'X', 'Y', and 'Z'.
        pauli_string = error_generator[1][0]
        # The following commented-out line is *wrong* but it used to be in the code, so leaving it here 
        # but commented out for now. It is unclear if somehow this was the correct thing to do in older
        # versions of the QPANN code before my (Tim's) rewrite.
        # pauli_string = pauli_string[::-1] # for reverse indexing
        #
        # The indices of `pauli` that are not equal to 'I' are the qubits that the error acts on
        qubits_acted_on_by_error = list(_np.where(_np.array(list(pauli_string)) != 'I')[0])
        # All the qubits that are within `hops` steps on the graph of the qubits acted on by the error
        relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
        # The encoding indices that encode what is happening to these qubits
        relevant_encoding_indices = encoder.indices_for_qubits(list(relevant_qubits))
        # Add to the list specifying the relevant encoding indices for each error generator in `error_generators`
        encoding_indices.append(relevant_encoding_indices)

    return encoding_indices
