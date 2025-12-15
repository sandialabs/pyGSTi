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
    TODO
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
    # Compute the set of qubits that are within `hops` steps on the adjacency graph of each qubit.
    degree_matrix = _np.diag(_np.sum(adjacency_matrix, axis = 1))
    laplacian = degree_matrix - adjacency_matrix
    laplace_power = _np.linalg.matrix_power(laplacian, hops)
    nodes_within_hops = [list(_np.arange(encoder.pspec.num_qubits)[abs(laplace_power[i, :]) > 0]) for i in range(encoder.pspec.num_qubits)]

    encoding_indices = []
    for error_generator in error_generators:
        # The Pauli that labels the error gen, as a string of length num_qubits containing 'I', 'X', 'Y', and 'Z'.
        pauli_string = error_generator[1][0]
        # TODO : IS THIS CORRECT ?
        # I COMMENTED THIS OUT ON THE RE-WRITE, BECAUSE I THINK IT'S WRONG. NEED TO CHECK I HAVEN'T MADE A 
        # MISTAKE THOUGH.
        pauli_string = pauli_string[::-1] # for reverse indexing

        # The indices of `pauli` that are not equal to 'I'.
        qubits_acted_on_by_error = _np.where(_np.array(list(pauli_string)) != 'I')[0]
        qubits_acted_on_by_error = list(qubits_acted_on_by_error)

        # All the qubits that are within `hops` steps of the qubits acted on by the error
        relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))

        relevant_encoding_indices_for_error_generator = encoder.indices_for_qubits(relevant_qubits)

        encoding_indices.append(relevant_encoding_indices_for_error_generator)

    return encoding_indices

# FUTURE TODO : CONSIDER IF THESE NEED TO BE REVIVED OR NOT        
# @_keras.utils.register_keras_serializable()
# def layer_snipper_from_qubit_graph_with_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
#                                                  num_hops, lookback=-1):
#     """

#     """
#     encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
#                                                                 num_channels=num_channels, 
#                                                                 qubit_graph_laplacian= qubit_graph_laplacian,
#                                                                 num_hops=num_hops)

#     indices_for_error = []
#     for relative_layer_index in range(lookback, 1):
#         indices_for_error += [[relative_layer_index, i] for i in encoding_indices_for_error]

#     return _np.array(indices_for_error)


# # @_keras.utils.register_keras_serializable()
# def layer_snipper_from_qubit_graph_with_simplified_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
#                                                  num_hops, lookback=-1):
#     """

#     """
#     encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
#                                                                 num_channels=num_channels, 
#                                                                 qubit_graph_laplacian= qubit_graph_laplacian,
#                                                                 num_hops=num_hops)

#     indices_for_error = []
#     for i in range(0, _np.abs(lookback)):
#         indices_for_error += list(encoding_indices_for_error + ((num_qubits * num_channels) * i))

#     return _np.array(indices_for_error)
    