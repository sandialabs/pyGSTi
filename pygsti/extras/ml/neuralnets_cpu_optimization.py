import tensorflow as _tf
import keras as _keras
import numpy as _np
import copy as _copy
import warnings

import tensorflow as tf
from tensorflow import keras as _keras
import copy as _copy


import tensorflow as tf
from tensorflow import keras as _keras
import copy as _copy


@_keras.utils.register_keras_serializable(package='Blah1')
class CircuitErrorVec(_keras.Model):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, 
                 layer_snipper, layer_snipper_args: list,
                 dense_units=[30, 20, 10, 5, 5], input_shape=None, **kwargs):
        """
        num_qubits: int
            The number of qubits that this neural network models.

        num_channels: int
            The number of gate channels in the tensor encoding of the circuits whose fidelity this network
            predicts.

        tracked_error_gens: list
            The primitive error generators that this neural network internally models.

        layer_snipper: func
            A function that takes a primitive error generator and maps it to a list that encodes which parts
            of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
            of that primitive error generator.

        input_shape: () not optional

        
        dense_units: list
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = _copy.deepcopy(tracked_error_gens)
        self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.dense_units = dense_units
        self.layer_snipper = layer_snipper
        self.layer_snipper_args = layer_snipper_args
        # self.input_layer = _keras.layers.InputLayer(shape=input_shape)
        self.hamiltonian_mask = tf.constant([1 if error[0] == 'H' else 0 for error in tracked_error_gens], tf.int32)
        self.stochastic_mask = tf.constant([1 if error[0] == 'S' else 0 for error in tracked_error_gens], tf.int32)

    def get_config(self):
        config = super(CircuitErrorVec, self).get_config()
        config.update({
            'num_qubits': self.num_qubits,
            'tracked_error_gens': self.tracked_error_gens,
            'num_channels': self.num_channels,
            'dense_units': self.dense_units,
            'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
            'layer_snipper_args': self.layer_snipper_args,
        })
        return config
    
    def build(self):
        self.local_dense = LocalizedDenseToErrVec(self.layer_snipper, self.layer_snipper_args, self.tracked_error_gens, self.dense_units)

    # @tf.function
    def calc_masked_err_rates(self, signed_M, P, mask):
        masked_M = tf.math.multiply(tf.cast(mask, tf.float32), signed_M)
        # print(mask.dtype, P.dtype)
        masked_P = tf.math.multiply(mask, P)
        flat_masked_M, flat_masked_P = tf.reshape(masked_M, [-1]), tf.reshape(masked_P, [-1])
        unique_masked_P, idx = tf.unique(flat_masked_P) # will come back to this. Possibly move to a model input rather than an intermediate calculation. Takes lots of time...
        num_segments = tf.reduce_max(idx) + 1
        # print(tf.math.unsorted_segment_sum(flat_masked_M, idx, num_segments).shape)
        return tf.math.unsorted_segment_sum(flat_masked_M, idx, num_segments)


    # @tf.function
    def calc_end_of_circ_error(self, M, P, S):
        """
        A function that maps the error rates (M) to an end-of-circuit error generator
        using the permutation matrix P.
        """
        # print('S, M', S.shape, M.shape)
        signed_M = tf.math.multiply(S, M) # mult alpha by M, assuming correct organization of alpha
        final_stochastic_error_rates = self.calc_masked_err_rates(signed_M, P, self.stochastic_mask)
        final_hamiltonian_error_rates = self.calc_masked_err_rates(signed_M, P, self.hamiltonian_mask)
        # print(final_stochastic_error_rates.shape, final_hamiltonian_error_rates.shape)
        return tf.reduce_sum(final_stochastic_error_rates) + tf.reduce_sum(tf.square(final_hamiltonian_error_rates))
    
    # @tf.function
    def circuit_to_fidelity(self, input): # will replace this with circ to probability dist
        """
        A function that maps a single circuit to the prediction for its process fidelity (a single real number).
        """
        # print('l', input)
        # C, P, S = input

        # print(C.shape, P.shape, S.shape)
        C = input[:, 0:self.len_gate_encoding]
        P = tf.cast(input[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], tf.int32)
        S = input[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
        
        M = self.local_dense(C)  # 
        
        return self.calc_end_of_circ_error(M, P, S)

    # @tf.function
    def call(self, inputs):
        return tf.map_fn(self.circuit_to_fidelity, inputs) # will try to further optimize this with batching rather than slow map_fn / vectorize_map
        


class DenseSubNetwork(_keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # self.outdim = outdim
        self.units = units
        self.outdim = units[-1]

    def build(self, input_shape):
        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential([
            _keras.layers.Dense(i, activation='gelu') for i in self.units[:-1]
        ])
        self.output_layer = _keras.layers.Dense(
            self.units[-1],
            kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001)
        )

    def get_config(self):
        config = super(DenseSubNetwork, self).get_config()
        config.update({
            'outdim': self.outdim,
            'units': self.units
        })
        return config

    def call(self, inputs):
        # This should naturally handle batches
        inputs = self.sequential(inputs)
        return self.output_layer(inputs)

# @_keras.utils.register_keras_serializable()
class LocalizedDenseToErrVec(_keras.layers.Layer):
    def __init__(self, layer_snipper, layer_snipper_args, tracked_error_gens, dense_units = [30, 20, 10, 5, 5], **kwargs):
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
        self.dense_units = dense_units + [self.num_tracked_error_gens]
        self.layer_snipper = layer_snipper
        self.layer_snipper_args = layer_snipper_args

    def get_config(self):
        config = super(LocalizedDenseToErrVec, self).get_config()
        config.update({
            'tracked_error_gens': self.tracked_error_gens,
            'dense_units': self.dense_units,
            'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
            'layer_snipper_args': self.layer_snipper_args
        })
        return config
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, input_shape[0], self.num_tracked_error_gens)

    @classmethod
    def from_config(cls, config):
        layer_snipper_config = config.pop("layer_snipper")
        layer_snipper = _keras.utils.deserialize_keras_object(layer_snipper_config)
        return cls(layer_snipper, **config)
    
    def build(self, input_shape):
        # self.dense = {error_gen_idx: DenseSubNetwork(1, self.dense_units) for error_gen_idx in range(self.num_tracked_error_gens)}
        self.dense = DenseSubNetwork(self.dense_units)
        super().build(input_shape)
    def call(self, inputs):
        # Convert the list of indices to a tensor
        indices_tensor = tf.stack(self.layer_encoding_indices_for_error_gen)
        
        # Expand dimensions to match the batch size
        batch_size = tf.shape(inputs)[0]
        indices_tiled = tf.tile(tf.expand_dims(indices_tensor, 0), [batch_size, 1, 1])

        # Gather the values based on the indices
        gathered_slices = tf.gather(inputs, indices_tiled, batch_dims=1)

        # Reshape the gathered slices to concatenate along the last axis
        gathered_slices = tf.reshape(gathered_slices, [batch_size, -1])
        
        # Pass the concatenated slices through the dense layer
        x = self.dense(gathered_slices)

        return x

# @_keras.utils.register_keras_serializable()
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
    pauli_string = pauli_string[::-1] # for reverse indexing
    # The indices of `pauli` that are not equal to 'I'.
    qubits_acted_on_by_error = _np.where(_np.array(list(pauli_string)) != 'I')[0]
    qubits_acted_on_by_error = list(qubits_acted_on_by_error)

    # All the qubits that are within `hops` steps, of the qubits acted on by the error, on the connectivity
    # graph of the qubits
    relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
    indices_for_error = _np.concatenate([[num_channels * q + i for i in range(num_channels)] for q in relevant_qubits])

    return indices_for_error
  