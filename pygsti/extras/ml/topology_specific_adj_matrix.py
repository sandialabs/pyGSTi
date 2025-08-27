import numpy as np

def grid_adj_matrix(grid_width: int):
    num_qubits = grid_width**2
    adj_matrix = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        if i % grid_width == grid_width - 1 and i != grid_width*grid_width - 1:
            # far right column, not the bottom left corner
            # print(i)
            # print('first')
            adj_matrix[i, i+grid_width] = 1
        elif i // grid_width == grid_width - 1 and i != grid_width*grid_width - 1:
            # bottom row, not the bottom left corner
            adj_matrix[i, i+1] = 1
        elif i != num_qubits - 1:
            # print(i)
            # print('third')
            # not the bottom right corner
            adj_matrix[i, i+grid_width] = 1
            adj_matrix[i, i+1] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return  adj_matrix

def ring_adj_matrix(num_qubits: int):
    adj_matrix = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        adj_matrix[i, (i-1) % num_qubits] = 1
        adj_matrix[i, (i+1) % num_qubits] = 1
    return adj_matrix

def melbourne_adj_matrix(num_qubits = 14):
    assert(num_qubits == 14), "We only support 5 qubits"
    adj_matrix = np.zeros((num_qubits, num_qubits))
    adj_matrix[0,1] = 1
    adj_matrix[1,2], adj_matrix[1,13] = 1,1
    adj_matrix[2,3], adj_matrix[2,12] = 1,1
    adj_matrix[3,4], adj_matrix[3,11] = 1,1
    adj_matrix[4,5], adj_matrix[4,10] = 1,1
    adj_matrix[5,6], adj_matrix[5,9] = 1,1
    adj_matrix[6,8] = 1
    adj_matrix[7,8] = 1
    adj_matrix[8,9] = 1
    adj_matrix[9,10] = 1
    adj_matrix[10,11] = 1
    adj_matrix[11,12] = 1
    adj_matrix[12,13] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def bowtie_adj_matrix(num_qubits = 5):
    '''
    Builds the adjacency matrix for a five-qubit bowtie graph:

    0 - 1
     \ /
      2
     / \
    3 - 4 
    '''
    assert(num_qubits == 5), "We only support 5 qubits"
    adj_matrix = np.zeros((num_qubits, num_qubits))
    adj_matrix[0, 1], adj_matrix[0, 2] = 1, 1
    adj_matrix[1, 2] = 1
    adj_matrix[2, 3], adj_matrix[2, 4] = 1, 1
    adj_matrix[3, 4] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def t_bar_adj_matrix(num_qubits = 5):
    '''
    Builds the adjacency matrix for a five-qubit T-bar graph:

    0 - 1 - 2
        |
        3
        |
        4
    '''
    assert(num_qubits == 5), "We only support 5 qubits"
    adj_matrix = np.zeros((num_qubits, num_qubits))
    adj_matrix[0, 1] = 1
    adj_matrix[1, 2], adj_matrix[1, 3] = 1, 1
    adj_matrix[3, 4] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def algiers_t_bar_adj_matrix():
    '''
    Builds the adjacency matrix for a five-qubit T-bar graph:

    0 - 1 - 4
        |
        2
        |
        3
    '''
    adj_matrix = np.zeros((5,5))
    adj_matrix[0, 1] = 1
    adj_matrix[1, 2], adj_matrix[1, 4] = 1, 1
    adj_matrix[2, 3] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix