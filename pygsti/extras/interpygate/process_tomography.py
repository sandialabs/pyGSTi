""" Perform process tomography on a simulated quantum process. """

import itertools as _itertools
from functools import reduce

import numpy as _np
import numpy.linalg as _lin

from pygsti.tools.basistools import change_basis
from pygsti.tools.legacytools import deprecate


#Helper functions
def multi_kron(*a):
    """ Construct the tensor product of a series of matrices """
    return reduce(_np.kron, a)


@deprecate("Calls to this function should be replaced with in-lined code: matrix.reshape((matrix.size, 1), 'F')")
def vec(matrix):
    """
    Returns an explicit column-vector representation of a square matrix, obtained by reading
    from the square matrix in column-major order.

    Args:
        matrix (list,numpy.ndarray): NxN matrix

    Returns:
        numpy.ndarray: N^2x1 dimensional column vector

    Raises:
        ValueError: If the input matrix is not square.

    """
    matrix = _np.array(matrix)
    if matrix.shape == (len(matrix), len(matrix)):
        return _np.array([_np.concatenate(_np.array(matrix).T)]).T
    else:
        raise ValueError('The input matrix must be square.')


@deprecate("Calls to this function should be replaced by unvec_square(vectorized, 'F')")
def unvec(vectorized):
    """A function that vectorizes a process in the basis of matrix units, sorted first
    by column, then row.

    Args:
        vectorized (list,numpy.ndarray): Nx1 matrix or N-dimensional vector

    Returns:
        numpy.ndarray: NxN dimensional column vector

    Raises:
        ValueError: If the length of the input is not a perfect square

    """
    vectorized = _np.array(vectorized)
    length = int(_np.sqrt(max(vectorized.shape)))
    if len(vectorized) == length ** 2:
        return _np.reshape(vectorized, [length, length]).T
    else:
        msg = 'The input vector length must be a perfect square, but this input has length %d.' % len(vectorized)
        raise ValueError(msg)


def split(n, a):
    """ Divide list into n approximately equally sized chunks

    Args:
        n : int
            The number of chunks

        a : iterable
            The array to be divided into chunks

    Returns:
        numpy.ndarray
            The original data divided into n approximately equally sized chunks.
    """
    k, m = divmod(len(a), n)
    return _np.array(list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


def run_process_tomography(state_to_density_matrix_fn, n_qubits=1, comm=None,
                           verbose=False, basis='pp', time_dependent=False, opt_args=None):
    """
    A function to compute the process matrix for a quantum channel given a function
    that maps a pure input state to an output density matrix.

    Args:
        state_to_density_matrix_fn : (function: array -> array)
            The function that computes the output density matrix from an input pure state.

        n_qubits : (int, optional, default 1)
            The number of qubits expected by the function. Defaults to 1.

        comm : (MPI.comm object, optional)
            An MPI communicator object for parallel computation. Defaults to local comm.

        verbose : (bool, optional, default False)
            How much detail to send to stdout

        basis : (str, optional, default 'pp')
            The basis in which to return the process matrix

        time_dependent : (bool, optional, default False )
            If the process is time dependent, then expect the density matrix function to
            return a list of density matrices, one at each time point.

        opt_args : (dict, optional)
            Optional keyword arguments for state_to_density_matrix_fn

    Returns:
        numpy.ndarray
            The process matrix representation of the quantum channel in the basis
            specified by 'basis'. If 'time_dependent'=True, then this will be an array
            of process matrices.
    """
    if opt_args is None:
        opt_args = {}
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    if verbose:
        print('Running process tomography as %d of %d on %s.' %
              (comm.Get_rank(), comm.Get_size(), comm.Get_name()))

    # Define and preprocess the input test states
    one_qubit_states = _np.array([[1, 0], [0, 1], [1, 1], [1., 1.j]], dtype='complex')
    one_qubit_states = [state / _lin.norm(state) for state in one_qubit_states]
    states = _itertools.product(one_qubit_states, repeat=n_qubits)
    states = [multi_kron(*state) for state in states]
    in_density_matrices = [_np.outer(state, state.conj()) for state in states]
    in_states = _np.column_stack(list([rho.ravel(order='F') for rho in in_density_matrices]))
    my_states = split(size, states)[rank]
    if verbose:
        print("Process %d of %d evaluating %d input states." % (rank, size, len(my_states)))
    if time_dependent:
        my_out_density_matrices = [state_to_density_matrix_fn(state, **opt_args) for state in my_states]
    else:
        my_out_density_matrices = [[state_to_density_matrix_fn(state, **opt_args)] for state in my_states]

    # Assemble the outputs
    if comm is not None:
        gathered_out_density_matrices = comm.gather(my_out_density_matrices, root=0)
    else:
        gathered_out_density_matrices = [my_out_density_matrices]

    if rank == 0:
        # Postprocess the output states to compute the process matrix
        # Flatten over processors
        out_density_matrices = _np.array([y for x in gathered_out_density_matrices for y in x])
        # Sort the list by time
        out_density_matrices = _np.transpose(out_density_matrices, [1, 0, 2, 3])
        out_states = [_np.column_stack(list([rho.ravel(order='F') for rho in density_matrices_at_time]))
                      for density_matrices_at_time in out_density_matrices]
        process_matrices = [_np.dot(out_states_at_time, _lin.inv(in_states)) for out_states_at_time in out_states]
        process_matrices = [change_basis(process_matrix_at_time, 'col', basis)
                            for process_matrix_at_time in process_matrices]

        if not time_dependent:
            return process_matrices[0]
        else:
            return process_matrices
    else:
        # print(f'Rank {rank} returning NONE from comm {comm}.')
        return None
