""" Circuit snippers for use in QPANNs """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

def undirected_adjacency_matrix_from_edges(edges, qubit_labels):
    """
    Constructs the adjacency matrix for the graph with nodes given by
    `qubit_labels` and edges given by `edges.

    Parameters
    ----------
    edges : list
        List of tuples of edges, where each element of each edge is an element of `qubit_labels`

    qubit_labels : list
        The nodes of the graph

    Returns
    -------
    numpy array
    """
    adjacency_matrix = _np.zeros((len(qubit_labels), len(qubit_labels)), int)
    for edge in edges:
        adjacency_matrix[qubit_labels.index(edge[0]), qubit_labels.index(edge[1])] = 1
        adjacency_matrix[qubit_labels.index(edge[1]), qubit_labels.index(edge[0])] = 1
    return adjacency_matrix

# FUTURE TODO: This function assumes only H and S errors. Will need to update if/when
# code is updated to add in C and A errors.
def layer_snipper_from_qubit_graph(error_generators, encoder, adjacency_matrix, hops):
    """
    TODO
    """
    # Compute the set of qubits that are within `hops` steps on the adjacency graph of each qubit,
    # by computing the Lapalacian and taking its `hops` power.
    degree_matrix = _np.diag(_np.sum(adjacency_matrix, axis = 1))
    laplacian = degree_matrix - adjacency_matrix
    laplace_power = _np.linalg.matrix_power(laplacian, hops)
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
        relevant_encoding_indices = encoder.indices_for_qubits(relevant_qubits)
        # Add to the list specifying the relevant encoding indices for each error generator in `error_generators`
        encoding_indices.append(relevant_encoding_indices)

    return encoding_indices
