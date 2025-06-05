import tensorflow as tf
import keras as _keras
import numpy as _np
import copy as _copy
import warnings
from pygsti.extras.ml import custom_layers

class DenseSubNetwork(_keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.outdim = units[-1]

    def build(self, input_shape):

        kernel_regularizer = None#_keras.regularizers.L2(1E-4)  # Adjust the regularization factor as needed
        bias_regularizer = None# _keras.regularizers.L2(1E-4)    # Adjust the regularization factor as needed

        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential(
            [_keras.layers.Dense(i, activation='gelu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer) for i in self.units[:-1]] +
            [_keras.layers.Dense(self.units[-1], activation='relu', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])

    def get_config(self):
        config = super(DenseSubNetwork, self).get_config()
        config.update({
            'outdim': self.outdim,
            'units': self.units
        })
        return config

    def call(self, inputs):
        return self.sequential(inputs)


class DenseEinsum(_keras.layers.Layer):
    def __init__(self, units, layer_encoding_indices_for_error_gen):
        super().__init__()
        self.units = units
        self.outdim = units[-1]
        self.num_errorgens = len(layer_encoding_indices_for_error_gen)
        self.layer_encoding_indices_for_error_gen = layer_encoding_indices_for_error_gen

    def build(self, input_shape):

        kernel_regularizer = None#_keras.regularizers.L2(1E-4)  # Adjust the regularization factor as needed
        bias_regularizer = None# _keras.regularizers.L2(1E-4)    # Adjust the regularization factor as needed
        init = _keras.initializers.RandomUniform(minval=-0.0001, maxval=0.0001)

        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential(
            # [custom_layers.SelectiveDense(self.units[0], self.layer_encoding_indices_for_error_gen,  activation='gelu')] +
             [custom_layers.CustomDense(i, self.num_errorgens, activation='gelu') for i in self.units[:-1]] +
            [custom_layers.CustomDense(self.units[-1], self.num_errorgens, activation='linear', kernel_initializer=init, bias_initializer=init)])

    def get_config(self):
        config = super(DenseSubNetwork, self).get_config()
        config.update({
            'outdim': self.outdim,
            'units': self.units
        })
        return config

    def call(self, inputs):
        return self.sequential(inputs)
        

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
        self.dense = DenseSubNetwork(self.dense_units)        
        super().build(input_shape)
    
    def old_call(self, inputs):
        print('circuit_input', inputs.shape)
        max_len_gate_encoding = max([len(layer_encoding) for layer_encoding in self.layer_encoding_indices_for_error_gen])
        indices_tensor = tf.ragged.constant(self.layer_encoding_indices_for_error_gen).to_tensor(
            default_value=-1, 
            shape=[len(self.layer_encoding_indices_for_error_gen), max_len_gate_encoding]
        ) # If fewer gate encodings than num_qubits*num_channels, pad with -1 (illegal index)
        
        # Expand dimensions to match the batch size
        batch_size = tf.shape(inputs)[0]
        indices_tiled = tf.tile(tf.expand_dims(indices_tensor, 0), [batch_size, 1, 1])

        # Create a mask based on the padding (-1 in indices_tensor), so that outputs from these indices can be masked out
        mask = tf.not_equal(indices_tiled, -1)
        mask = tf.cast(mask, dtype=inputs.dtype)

        # Change -1 to 0 in indices_tiled before using tf.gather
        indices_tiled = tf.where(indices_tiled == -1, tf.zeros_like(indices_tiled), indices_tiled) # replace indices of -1 (error) to 0 (will point to the wrong index)
        # Gather the values based on the indices
        gathered_slices = tf.gather(inputs, indices_tiled, batch_dims=1)

        # Apply the mask to zero out the gathered slices at the padding positions
        gathered_slices_masked = gathered_slices * mask

        # Reshape the gathered slices to concatenate along the last axis
        gathered_slices_flat = tf.reshape(gathered_slices_masked, [batch_size, -1])

        # Dense network to learn error rates
        x = self.dense(gathered_slices_flat)
        return x

    def call_redundant(self, inputs):
        stacked_index = tf.concat(self.layer_encoding_indices_for_error_gen, axis=0)
        gathered_slices_flat = tf.gather(inputs, stacked_index, axis=1)
        x = self.dense(gathered_slices_flat)
        return x

    def call(self, inputs):
        x = self.dense(inputs)
        return x

# @_keras.utils.register_keras_serializable()
class GraphDenseToErrVec(_keras.layers.Layer):
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
        self.dense_units = dense_units + [1]
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
        self.dense = [DenseSubNetwork(self.dense_units) for error_gen_idx in range(self.num_tracked_error_gens)]
        super().build(input_shape)

    def call(self, inputs):
        x = [self.dense[i](tf.gather(inputs, self.layer_encoding_indices_for_error_gen[i], axis=-1)) 
             for i in range(0, self.num_tracked_error_gens)]
        x = tf.concat(x, axis=-1)
        return x

        

# @_keras.utils.register_keras_serializable()
def layer_snipper_from_qubit_graph(error_gen, num_qubits, num_channels, qubit_graph_laplacian, num_hops):
    """

    """

    laplace_power = _np.linalg.matrix_power(qubit_graph_laplacian, num_hops)
    nodes_within_hops = []
    for i in range(num_qubits):
        print(i)
        nodes_within_hops.append(_np.arange(num_qubits)[abs(laplace_power[i, :]) > 0])

    # These next few lines assumes only 'H' and 'S' errors because it only looks at the *first* Pauli that labels 
    # the error generator, but there are two Paulis for 'A' and 'C'
    
    # The Pauli that labels the error gen, as a string of length num_qubits containing 'I', 'X', 'Y', and 'Z'.
    pauli_string = error_gen[1][0]
    print(pauli_string)
    pauli_string = pauli_string[::-1] # for reverse indexing
    # The indices of `pauli` that are not equal to 'I'.
    qubits_acted_on_by_error = _np.where(_np.array(list(pauli_string)) != 'I')[0]
    qubits_acted_on_by_error = list(qubits_acted_on_by_error)

    # All the qubits that are within `hops` steps, of the qubits acted on by the error, on the connectivity
    # graph of the qubits
    relevant_qubits = _np.unique(_np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
    indices_for_error = _np.concatenate([[num_channels * q + i for i in range(num_channels)] for q in relevant_qubits])

    return indices_for_error

        
# @_keras.utils.register_keras_serializable()
def layer_snipper_from_qubit_graph_with_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
                                                 num_hops, lookback=-1):
    """

    """
    encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
                                                                num_channels=num_channels, 
                                                                qubit_graph_laplacian= qubit_graph_laplacian,
                                                                num_hops=num_hops)

    indices_for_error = []
    for relative_layer_index in range(lookback, 1):
        indices_for_error += [[relative_layer_index, i] for i in encoding_indices_for_error]

    return _np.array(indices_for_error)


# @_keras.utils.register_keras_serializable()
def layer_snipper_from_qubit_graph_with_simplified_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
                                                 num_hops, lookback=-1):
    """

    """
    encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
                                                                num_channels=num_channels, 
                                                                qubit_graph_laplacian= qubit_graph_laplacian,
                                                                num_hops=num_hops)

    indices_for_error = []
    for i in range(0 np.abs(lookback)):
        indices_for_error += list(encoding_indices_for_error) + (num_qubits * num_channels) * i)

    return _np.array(indices_for_error)


@_keras.utils.register_keras_serializable(package='Blah1')
class CircuitErrorVecMap(_keras.Model):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, 
                 layer_snipper, layer_snipper_args: list,
                 dense_units=[30, 20, 10, 5, 5], **kwargs):
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

        
        dense_units: list
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = _copy.deepcopy(tracked_error_gens)
        # self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.dense_units = dense_units
        self.layer_snipper = layer_snipper
        self.layer_snipper_args = layer_snipper_args

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
        self.dense_layer = LocalizedDenseToErrVec(self.layer_snipper, self.layer_snipper_args, self.tracked_error_gens, self.dense_units)
        self.probability_approximation_layer = EndOfCircProbabilityLayer()
        self.dense_correction = _keras.Sequential(
            [_keras.layers.Dense(i, activation='gelu') for i in self.dense_units] +
            [_keras.layers.Dense(32, activation='linear', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])
    
    # @tf.function
    # def calc_end_of_circ_error_rates(self, M, P, S, scaled_alpha_matrix):
    #     """
    #     A function that maps the error rates (M) to an end-of-circuit error generator
    #     using the permutation matrix P.
    #     """
    #     signed_M = tf.math.multiply(S, M) 
    #     flat_signed_M, flat_P = tf.reshape(signed_M, [-1]), tf.reshape(P, [-1])
    #     unique_P, idx = tf.unique(flat_P) # unique_P values [0, num_error_generators]
    #     num_segments = tf.reduce_max(idx)+1
    #     error_rates = tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
    #     gathered_alpha = tf.gather(scaled_alpha_matrix, unique_P, axis=1)
    #     first_order_correction = gathered_alpha*error_rates
    #     return first_order_correction

    # @tf.function
    # def circuit_to_probability(self, inputs): # will replace this with circ to probability dist
    #     """
    #     A function that maps a single circuit to the prediction of a 1st order approximate probability vector for each of 2^Q bitstrings.
    #     """  
        
    #     circuit_encoding = inputs[0] # circuit
    #     S = tf.cast(inputs[1], tf.float32) # sign matrix
    #     P = tf.cast(inputs[2], tf.int32) # permutation matrix
    #     scaled_alpha_matrix = inputs[3] # alphas (shape is number of tracked error gens, typically 132. Very sparse, could use sparse LA at a later time)
    #     Px_ideal = inputs[4] # ideal (no error) probabilities
        
    #     epsilon_matrix = self.local_dense(circuit_encoding) # depth * num_tracked_error
    #     first_order_correction = self.calc_end_of_circ_error_rates(epsilon_matrix, P, S, scaled_alpha_matrix)
    #     Px_approximate = tf.reduce_sum(first_order_correction, 1) + Px_ideal
    #     # Px_approximate_clipped = tf.reshape(tf.clip_by_value(Px_approximate, 0, 1), [32])
    #     return Px_approximate
    def circuit_to_probability(self, inputs):
        circuit_encoding = inputs[0]  # circuit
        S = tf.cast(inputs[1], tf.float32)  # sign matrix
        P = tf.cast(inputs[2], tf.int32)  # permutation matrix
        scaled_alpha_matrix = inputs[3]  # alphas
        Px_ideal = inputs[4]  # ideal (no error) probabilities
        # C = tf.reshape(self.dense_correction(tf.reshape(circuit_encoding, [1, -1])), [-1])
        epsilon_matrix = self.dense_layer(circuit_encoding)  # depth * num_tracked_error
        Px_approximate = self.probability_approximation_layer([epsilon_matrix, P, S, scaled_alpha_matrix, Px_ideal]) #+C

        return Px_approximate# / tf.reduce_sum(Px_approximate)

    # @tf.function
    def call(self, inputs):
        return tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=tf.float32)


@_keras.utils.register_keras_serializable(package='Blah1')
class CircuitErrorGraph(_keras.Model):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, 
                 layer_snipper, layer_snipper_args: list,
                 dense_units=[30, 20, 10, 5, 5], **kwargs):
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
        self.local_dense = GraphDenseToErrVec(self.layer_snipper, self.layer_snipper_args, self.tracked_error_gens, self.dense_units)
        # self.dense = _keras.Sequential(
        #     [_keras.layers.Dense(i, activation='gelu') for i in [2**5, 2**5]] +
        #     [_keras.layers.Dense([1], activation='sigmoid', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])
    
    # @tf.function
    def calc_end_of_circ_error_rates(self, M, P, S, scaled_alpha_matrix):
        """
        A function that maps the error rates (M) to an end-of-circuit error generator
        using the permutation matrix P.
        """
        signed_M = tf.math.multiply(S, M) 
        flat_signed_M, flat_P = tf.reshape(signed_M, [-1]), tf.reshape(P, [-1])
        unique_P, idx = tf.unique(flat_P) # unique_P values [0, num_error_generators]
        num_segments = tf.reduce_max(idx)+1
        error_rates = tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
        gathered_alpha = tf.gather(scaled_alpha_matrix, unique_P, axis=1)
        first_order_correction = gathered_alpha*error_rates
        return first_order_correction

    # @tf.function
    def circuit_to_probability(self, inputs): # will replace this with circ to probability dist
        """
        A function that maps a single circuit to the prediction of a 1st order approximate probability vector for each of 2^Q bitstrings.
        """  
        
        circuit_encoding = inputs[0] # circuit
        S = tf.cast(inputs[1], tf.float32) # sign matrix
        P = tf.cast(inputs[2], tf.int32) # permutation matrix
        scaled_alpha_matrix = inputs[3] # alphas (shape is number of tracked error gens, typically 132. Very sparse, could use sparse LA at a later time)
        Px_ideal = inputs[4] # ideal (no error) probabilities
        
        epsilon_matrix = self.local_dense(circuit_encoding) # depth * num_tracked_error
        first_order_correction = self.calc_end_of_circ_error_rates(epsilon_matrix, P, S, scaled_alpha_matrix)
        Px_approximate = tf.reduce_sum(first_order_correction, 1) + Px_ideal
        # Px_approximate_clipped = tf.reshape(tf.clip_by_value(Px_approximate, 0, 1), [32])
        return Px_approximate

    # @tf.function
    def call(self, inputs):
        output = tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=tf.float32)
        return output
        # print(output.shape, output.dtype, type(self.dense))
        # return self.dense(output)



class EndOfCircProbabilityLayer(_keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EndOfCircProbabilityLayer, self).__init__(**kwargs)
        self.bitstring_shape = None
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs):
        M, P, S, scaled_alpha_matrix, Px_ideal = inputs
        self.bitstring_shape = Px_ideal.shape[0]
        signed_M = tf.math.multiply(S, M) 
        flat_signed_M, flat_P = tf.reshape(signed_M, [-1]), tf.reshape(P, [-1])
        unique_P, idx = tf.unique(flat_P)  # unique_P values [0, num_error_generators]
        num_segments = tf.reduce_max(idx) + 1
        error_rates = tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
        gathered_alpha = tf.gather(scaled_alpha_matrix, unique_P, axis=1)
        first_order_correction = gathered_alpha * error_rates
        Px_approximate = tf.reduce_sum(first_order_correction, 1) + Px_ideal
        return Px_approximate

class EndOfCircFidelityLayer(_keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EndOfCircFidelityLayer, self).__init__(**kwargs)
        self.bitstring_shape = None
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs):
        M, P, S = inputs
        self.bitstring_shape = Px_ideal.shape[0]
        signed_M = tf.math.multiply(S, M) 
        flat_signed_M, flat_P = tf.reshape(signed_M, [-1]), tf.reshape(P, [-1])
        unique_P, idx = tf.unique(flat_P)  # unique_P values [0, num_error_generators]
        num_segments = tf.reduce_max(idx) + 1
        error_rates = tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
        gathered_alpha = tf.gather(scaled_alpha_matrix, unique_P, axis=1)
        first_order_correction = gathered_alpha * error_rates
        Px_approximate = tf.reduce_sum(first_order_correction, 1) + Px_ideal
        return Px_approximate


@_keras.utils.register_keras_serializable(package='Blah1')
class CircuitErrorEinsum(_keras.Model):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, 
                 layer_snipper, layer_snipper_args: list,
                 dense_units=[30, 20, 10, 5, 5], **kwargs):
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

        
        dense_units: list
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = _copy.deepcopy(tracked_error_gens)
        # self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.dense_units = dense_units
        self.layer_snipper = layer_snipper
        self.layer_snipper_args = layer_snipper_args

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
        self.dense_layer = LocalizedEinsumDenseToErrVec(self.layer_snipper, self.layer_snipper_args, self.tracked_error_gens, self.dense_units)
        self.probability_approximation_layer = EndOfCircProbabilityLayer()
        # self.dense_correction = _keras.Sequential(
        #     [_keras.layers.Dense(i, activation='gelu') for i in self.dense_units] +
        #     [_keras.layers.Dense(32, activation='tanh', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])
    
 
    def circuit_to_probability(self, inputs):
        circuit_encoding = inputs[0]  # circuit
        S = tf.cast(inputs[1], tf.float32)  # sign matrix
        P = tf.cast(inputs[2], tf.int32)  # permutation matrix
        scaled_alpha_matrix = inputs[3]  # alphas
        Px_ideal = inputs[4]  # ideal (no error) probabilities
        # C = tf.reshape(self.dense_correction(tf.reshape(circuit_encoding, [1, -1])), [-1])
        epsilon_matrix = self.dense_layer(circuit_encoding)  # depth * num_tracked_error
        Px_approximate = self.probability_approximation_layer([epsilon_matrix, P, S, scaled_alpha_matrix, Px_ideal]) #+C
        
        return Px_approximate / tf.reduce_sum(Px_approximate)

    # @tf.function
    def call(self, inputs):
        return tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=tf.float32)


class LocalizedEinsumDenseToErrVec(_keras.layers.Layer):
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
        self.dense_units = dense_units + [1]
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
        self.dense = DenseEinsum(self.dense_units, self.layer_encoding_indices_for_error_gen)        
        super().build(input_shape)
    
    def call(self, inputs):
        max_len_gate_encoding = max([len(layer_encoding) for layer_encoding in self.layer_encoding_indices_for_error_gen])
        indices_tensor = tf.ragged.constant(self.layer_encoding_indices_for_error_gen).to_tensor(
            default_value=-1, 
            shape=[len(self.layer_encoding_indices_for_error_gen), max_len_gate_encoding]
        ) # If fewer gate encodings than num_qubits*num_channels, pad with -1 (illegal index)
        
        # Expand dimensions to match the batch size
        batch_size = tf.shape(inputs)[0]
        indices_tiled = tf.tile(tf.expand_dims(indices_tensor, 0), [batch_size, 1, 1])

        # Create a mask based on the padding (-1 in indices_tensor), so that outputs from these indices can be masked out
        mask = tf.not_equal(indices_tiled, -1)
        mask = tf.cast(mask, dtype=inputs.dtype)

        # Change -1 to 0 in indices_tiled before using tf.gather
        indices_tiled = tf.where(indices_tiled == -1, tf.zeros_like(indices_tiled), indices_tiled) # replace indices of -1 (error) to 0 (will point to the wrong index)
        # Gather the values based on the indices
        gathered_slices = tf.gather(inputs, indices_tiled, batch_dims=1)

        # Apply the mask to zero out the gathered slices at the padding positions
        gathered_slices_masked = gathered_slices * mask

        print('gathered_slices_masked', gathered_slices_masked.shape)

        # Reshape the gathered slices to concatenate along the last axis
        gathered_slices_flat = tf.reshape(gathered_slices_masked, [batch_size, self.num_tracked_error_gens, -1])

        print('gathered_slices_flat', gathered_slices_flat.shape)

        # Dense network to learn error rates
        x = tf.reshape(self.dense(gathered_slices_masked), [-1, self.num_tracked_error_gens])
        return x
    # def call(self, inputs):
    #     x = tf.reduce_sum(self.dense(inputs), -1)
    #     return x
        
        
        