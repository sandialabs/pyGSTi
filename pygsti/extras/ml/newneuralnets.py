import tensorflow as _tf
import keras as _keras
import numpy as _np
# from . import tools as qcl
import copy as _copy


class DenseSubNetwork(_keras.layers.Layer):
    def __init__(self, outdim):
        super().__init__()
        self.outdim = outdim
        self.output_layer = _keras.layers.Dense(outdim, kernel_initializer=_keras.initializers.random_uniform(minval=-0.00001, maxval=0.00001))

    def build(self, input_shape):
        # Define the sub-unit's dense layers
        self.dense1 = _keras.layers.Dense(30, activation='relu')
        self.dense2 = _keras.layers.Dense(20, activation='relu')
        self.dense3 = _keras.layers.Dense(10, activation='relu')
        super().build(input_shape)

    def call(self, inputs):
        # This should naturally handle batches....
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


def layer_snipper_from_qubit_graph(error_gen, num_qubits, num_channels, qubit_graph_laplacian, num_hops):
    """

    """

    laplace_power = _np.linalg.matrix_power(qubit_graph_laplacian, num_hops)
    nodes_within_hops = []
    for i in range(num_qubits):
        nodes_within_hops.append(_np.arange(num_qubits)[abs(laplace_power[i, :]) > 0])

    # These next few lines assumes only 'H' and 'S' errors because it only looks at the *first* Pauli that labels 
    # the error generator, but there are two Paulis for 'A' and 'C'
    
    # The Pauli that labels the error gen, as a string of length num_qubits containing 'I', 'X', 'Y', and 'Z'.
    pauli_string = error_gen[1][0]
    # The indices of `pauli` that are not equal to 'I'.
    qubits_acted_on_by_error = list(_np.where(_np.array(list(pauli_string)) != 'I')[0])
    # All the qubits that are within `hops` steps, of the qubits acted on by the error, on the connectivity
    # graph of the qubits
    relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))

    indices_for_error = _np.concatenate([[num_channels * q + i for i in range(num_channels)] for q in relevant_qubits])

    return indices_for_error
    

class LocalizedDenseToErrVec(_keras.layers.Layer):
    def __init__(self, layer_snipper, layer_snipper_args, tracked_error_gens):
        """
        layer_snipper: func
            A function that takes a primitive error generator and maps it to a list that encodes which parts
            of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
            of that primitive error generator.

        tracked_error_gens: list
            A list of the primitive error generators that are to be predicted.
        """
        super().__init__()
        
        self.num_tracked_error_gens = len(tracked_error_gens)  # This is the output dimenision of the network
        self.tracked_error_gens = tracked_error_gens  
        self.layer_encoding_indices_for_error_gen = [layer_snipper(error_gen, *layer_snipper_args) for error_gen in tracked_error_gens]

    def build(self, input_shape):
        self.dense = {error_gen_idx: DenseSubNetwork(1) for error_gen_idx in range(self.num_tracked_error_gens)}
        super().build(input_shape)

    def call(self, inputs):
        x = [self.dense[i](_tf.gather(inputs, self.layer_encoding_indices_for_error_gen[i], axis=-1)) 
             for i in range(0, self.num_tracked_error_gens)]
        x = _tf.concat(x, axis=-1)
        return x


class CircuitErrorVec(_keras.Model):
    def __init__(self, num_qubits, num_channels, tracked_error_gens, layer_snipper, layer_snipper_args, 
                 input_shape=None):
        """
        num_qubits: int
            The number of qubits that this neural network models.

        num_channels: int
            The number of gate channels in the tensor encoding of the circuits whose fidelity this network
            predicts

        tracked_error_gens: list
            The primitive error generators that this neural network internally models.

        layer_snippper: func
            A function that takes a primitive error generator and maps it to a list that encodes which parts
            of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
            of that primitive error generator.

        input_shape: optional
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = _copy.deepcopy(tracked_error_gens)
        self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.input_layer = _keras.layers.InputLayer(input_shape=input_shape)
        self.local_dense = LocalizedDenseToErrVec(layer_snipper, layer_snipper_args, self.tracked_error_gens)
   
    def call(self, inputs):
        # This is very slow when it is called on a large number of circuits. It's because it is not implemented
        # as efficiently (map_fn is the slow part)
        # But that may not be an issue if you keep the batch sizes smallish
        def calc_end_of_circ_err_vec(M, P, S):
            """
            A function that maps the error rates (M) to an end-of-circuit error generator
            using the permutation matrix P.
            """
            signed_M = _tf.math.multiply(S, M)
            flat_M, flat_P = _tf.reshape(signed_M, [-1]), _tf.reshape(P, [-1])
            
            num_segments = _tf.reduce_max(flat_P) + 1
            return _tf.math.unsorted_segment_sum(flat_M, flat_P, num_segments)  # This returns a larger vector than necessary

        def calc_fidelity(final_evec):
            # TO DO: This needs to be generalized when we include 'S' errors.
            return _tf.reduce_sum(final_evec**2, axis=-1)

        def circuit_to_fidelity(input):
            """
            A function that maps a single circuit to the prediction for its process fidelity (a single real number).
            """
            C = input[:, 0:self.len_gate_encoding]
            P = _tf.cast(input[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], _tf.int32)
            S = input[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
            evecs = self.local_dense(self.input_layer(C))
            total_evec = calc_end_of_circ_err_vec(evecs, P, S)
            return calc_fidelity(total_evec)
        
        return _tf.map_fn(circuit_to_fidelity, inputs)   
