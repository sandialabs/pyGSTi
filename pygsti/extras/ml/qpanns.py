""" Quantum physics aware neural networks (QPANNs) """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import tensorflow as _tf
import keras as _keras
import numpy as _np
import copy as _copy
import warnings as _warnings
from pygsti.extras.ml import customlayers as _cl

# TIM HAS NOT UPDATED THESE VERSIONS OF THE QPANNs TO THE NEW CODE
# class DenseSubNetwork(_keras.layers.Layer):
#     def __init__(self, units):
#         super().__init__()
#         self.units = units
#         self.outdim = units[-1]

#     def build(self, input_shape):

#         kernel_regularizer = None#_keras.regularizers.L2(1E-4)  # Adjust the regularization factor as needed
#         bias_regularizer = None# _keras.regularizers.L2(1E-4)    # Adjust the regularization factor as needed

#         # Define the sub-unit's dense layers
#         self.sequential = _keras.Sequential(
#             [_keras.layers.Dense(i, activation='gelu', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer) for i in self.units[:-1]] +
#             [_keras.layers.Dense(self.units[-1], activation='relu', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])

#     def get_config(self):
#         config = super(DenseSubNetwork, self).get_config()
#         config.update({
#             'outdim': self.outdim,
#             'units': self.units
#         })
#         return config

#     def call(self, inputs):
#         return self.sequential(inputs)


class EinsumSubNetwork(_keras.layers.Layer):
    def __init__(self, units, snipper):
        super().__init__()
        self.units = units
        self.outdim = units[-1]
        self.number_of_modelled_error_generators = len(snipper)
        self.snipper = snipper

    def build(self, input_shape):

        kernel_regularizer = None#_keras.regularizers.L2(1E-4)  # Adjust the regularization factor as needed
        bias_regularizer = None# _keras.regularizers.L2(1E-4)    # Adjust the regularization factor as needed
        init = _keras.initializers.RandomUniform(minval=-0.0001, maxval=0.0001)

        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential(
            # [_cl.SelectiveDense(self.units[0], self.layer_encoding_indices_for_error_generators,  activation='gelu')] +
             [_cl.CustomDense(i, self.number_of_modelled_error_generators, activation='linear') for i in self.units[:-1]] +
            [_cl.CustomDense(self.units[-1], self.number_of_modelled_error_generators, activation='linear', kernel_initializer=init, bias_initializer=init)])

    def get_config(self):
        config = super().get_config()
        config.update({
            'outdim': self.outdim,
            'units': self.units
        })
        return config

    def call(self, inputs):
        return self.sequential(inputs)
        

@_keras.utils.register_keras_serializable(package='Blah1')
class QPANN(_keras.Model):
    def __init__(self, encoding_length : int, modelled_error_generators: list,  snipper: list,
                 dense_units=[30, 20, 10, 5, 5], probability_computation='concise', **kwargs):
        """

        modelled_error_generators: list
            The primitive error generators that this neural network internally models.

        layer_snipper: ...

        
        dense_units: list
        """
        super().__init__()
        self.encoding_length = encoding_length
        self.modelled_error_generators = _copy.deepcopy(modelled_error_generators)
        self.snipper = _copy.deepcopy(snipper)
        self.dense_units = dense_units
        self.probability_computation = probability_computation
        # A mask that finds the 'S' (stochastic) error generators.
        self.stochastic_mask = _tf.constant([i[0] == 'S' for i in self.modelled_error_generators])
        self.hamiltonian_mask = _tf.constant([i[0] == 'H' for i in self.modelled_error_generators])

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoding_length': self.encoding_length,
            'modelled_error_generators': self.modelled_error_generators,
            'snipper': self.snipper,
            'dense_units': self.dense_units,
            'probability_computation': self.probability_computation,
            'stochastic_mask': self.stochastic_mask,
            'hamiltonian_mask': self.hamiltonian_mask
        })
        return config

    def build(self):
        self.dense_layer = CircuitToErrorRatesEinSum(self.snipper, self.modelled_error_generators, self.dense_units)
        if self.probability_computation == 'expanded':
            self.probability_approximation_layer = ProbabilitiesLayer()
        elif self.probability_computation == 'concise':
            self.probability_approximation_layer = ProbabilitiesLayerConcise()

    def circuit_to_probability(self, inputs):
        circuit_encoding = inputs[0]  # circuit

        # C = _tf.reshape(self.dense_correction(_tf.reshape(circuit_encoding, [1, -1])), [-1])
        epsilon_matrix = self.dense_layer(circuit_encoding)  # depth * num_tracked_error

        # # Define the function to apply
        # def custom_function(row):
        #     return row ** 2

        # # Expand the mask to match the tensor's shape for broadcasting
        # mask_expanded = _tf.expand_dims(_tf.expand_dims(self.stochastic_mask, axis=0), axis=0)

        # # Apply the function conditionally using _tf.where
        # # If mask_expanded is True, apply custom_function, otherwise keep original row
        # s_squared_epsilon_matrix = _tf.where(mask_expanded, custom_function(epsilon_matrix), epsilon_matrix)

        s_squared_epsilon_matrix = epsilon_matrix

        if self.probability_computation == 'expanded':
            S = _tf.cast(inputs[1], _tf.float32)  # sign matrix
            P = _tf.cast(inputs[2], _tf.int32)  # permutation matrix
            scaled_alpha_matrix = inputs[3]  # alphas
            probabilities_ideal = inputs[4]  # ideal (no error) probabilities
            probabilities = self.probability_approximation_layer([s_squared_epsilon_matrix, P, S, scaled_alpha_matrix, probabilities_ideal]) #+C

        elif self.probability_computation == 'concise':
            corrections_coefficients = inputs[1]  # alphas
            probabilities_ideal = inputs[2]  # ideal (no error) probabilities
            probabilities= self.probability_approximation_layer([s_squared_epsilon_matrix, corrections_coefficients, probabilities_ideal]) #+C

        #Px_approximate = self.probability_approximation_layer([epsilon_matrix, P, S, scaled_alpha_matrix, Px_ideal]) #+C
        
        return probabilities #/ _tf.reduce_sum(Px_approximate)

    # @_tf.function
    def call(self, inputs):
        return _tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=_tf.float32)

# TIM HAS NOT FULLY UPDATED THESE VERSIONS OF THE QPANNs TO THE NEW CODE
# # @_keras.utils.register_keras_serializable()
# class DenseToErrVec(_keras.layers.Layer):
#     def __init__(self, snipper, modelled_error_generators, dense_units = [30, 20, 10, 5, 5], **kwargs):
#         """
#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

#         modelled_error_generators: list
#             A list of the primitive error generators that are to be predicted.
#         """
#         super().__init__()
        
#         self.number_of_modelled_error_generators = len(modelled_error_generators)  # This is the output dimenision of the network
#         self.modelled_error_generators = modelled_error_generators  
#         self.snipper = snipper
#         self.dense_units = dense_units + [self.number_of_modelled_error_generators]

#     def get_config(self):
#         config = super(LocalizedDenseToErrVec, self).get_config()
#         config.update({
#             'modelled_error_generators': self.modelled_error_generators,
#             'dense_units': self.dense_units,
#             'snipper': self.snipper,
#         })
#         return config
    
#     def compute_output_shape(self, input_shape):
#         # Define the output shape based on the input shape and the number of tracked error generators
#         return (None, input_shape[0], self.number_of_modelled_error_generators)

#     # @classmethod
#     # def from_config(cls, config):
#     #     layer_snipper_config = config.pop("layer_snipper")
#     #     layer_snipper = _keras.utils.deserialize_keras_object(layer_snipper_config)
#     #     return cls(layer_snipper, **config)
    
#     def build(self, input_shape):
#         self.dense = DenseSubNetwork(self.dense_units)        
#         super().build(input_shape)
    
#     def call(self, inputs):
#         x = self.dense(inputs)
#         return x

# # @_keras.utils.register_keras_serializable()
# class GraphToErrVec(_keras.layers.Layer):
#     def __init__(self, layer_snipper, layer_snipper_args, modelled_error_generators, dense_units = [30, 20, 10, 5, 5], **kwargs):
#         """
#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

#         modelled_error_generators: list
#             A list of the primitive error generators that are to be predicted.
#         """
#         super().__init__()
        
#         self.number_of_modelled_error_generators = len(modelled_error_generators)  # This is the output dimenision of the network
#         self.modelled_error_generators = modelled_error_generators  
#         self.layer_encoding_indices_for_error_generators = [layer_snipper(error_gen, *layer_snipper_args) for error_gen in modelled_error_generators]
#         self.dense_units = dense_units + [1]
#         self.layer_snipper = layer_snipper
#         self.layer_snipper_args = layer_snipper_args

#     def get_config(self):
#         config = super(LocalizedDenseToErrVec, self).get_config()
#         config.update({
#             'modelled_error_generators': self.modelled_error_generators,
#             'dense_units': self.dense_units,
#             'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
#             'layer_snipper_args': self.layer_snipper_args
#         })
#         return config
    
#     def compute_output_shape(self, input_shape):
#         # Define the output shape based on the input shape and the number of tracked error generators
#         return (None, input_shape[0], self.number_of_modelled_error_generators)

#     @classmethod
#     def from_config(cls, config):
#         layer_snipper_config = config.pop("layer_snipper")
#         layer_snipper = _keras.utils.deserialize_keras_object(layer_snipper_config)
#         return cls(layer_snipper, **config)
    
#     def build(self, input_shape):
#         # self.dense = {error_gen_idx: DenseSubNetwork(1, self.dense_units) for error_gen_idx in range(self.number_of_modelled_error_generators)}
#         self.dense = [DenseSubNetwork(self.dense_units) for error_gen_idx in range(self.number_of_modelled_error_generators)]
#         super().build(input_shape)

#     def call(self, inputs):
#         x = [self.dense[i](_tf.gather(inputs, self.layer_encoding_indices_for_error_generators[i], axis=-1)) 
#              for i in range(0, self.number_of_modelled_error_generators)]
#         x = _tf.concat(x, axis=-1)
#         return x


# TIM HAS NOT UPDATED THIS VERSION OF THE QPANNs FOR THE NEW CODE. THE OBJECTS COMMENTED OUT JUST BELOW ARE
# ALTERNATIVE VERSIONS OF THE TOP-LEVEL QPANN OBJECT.
# @_keras.utils.register_keras_serializable(package='Blah1')
# class DenseCircuitErrorVec(_keras.Model):
#     def __init__(self, num_qubits: int, encoding_length: int, modelled_error_generators: list, 
#                  layer_snipper, layer_snipper_args: list,
#                  dense_units=[30, 20, 10, 5, 5], **kwargs):
#         """
#         num_qubits: int
#             The number of qubits that this neural network models.

#         encoding_length: int
#             The number of gate channels in the tensor encoding of the circuits whose fidelity this network
#             predicts.

#         modelled_error_generators: list
#             The primitive error generators that this neural network internally models.

#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

        
#         dense_units: list
#         """
#         super().__init__()
#         self.num_qubits = num_qubits
#         self.modelled_error_generators = _copy.deepcopy(modelled_error_generators)
#         # self.number_of_modelled_error_generators = len(self.modelled_error_generators)
#         self.encoding_length = encoding_length
#         self.dense_units = dense_units
#         self.layer_snipper = layer_snipper
#         self.layer_snipper_args = layer_snipper_args

#     def get_config(self):
#         config = super(CircuitErrorVec, self).get_config()
#         config.update({
#             'num_qubits': self.num_qubits,
#             'modelled_error_generators': self.modelled_error_generators,
#             'encoding_length': self.encoding_length,
#             'dense_units': self.dense_units,
#             'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
#             'layer_snipper_args': self.layer_snipper_args,
#         })
#         return config

#     def build(self):
#         self.dense_layer = DenseToErrVec(self.layer_snipper, self.layer_snipper_args, self.modelled_error_generators, self.dense_units)
#         self.probability_approximation_layer = ProbabilitiesLayer()
#         self.dense_correction = _keras.Sequential(
#             [_keras.layers.Dense(i, activation='gelu') for i in self.dense_units] +
#             [_keras.layers.Dense(32, activation='linear', kernel_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001), bias_initializer=_keras.initializers.RandomUniform(minval=-0.00001, maxval=0.00001))])
    

#     def circuit_to_probability(self, inputs):
#         circuit_encoding = inputs[0]  # circuit
#         S = _tf.cast(inputs[1], _tf.float32)  # sign matrix
#         P = _tf.cast(inputs[2], _tf.int32)  # permutation matrix
#         scaled_alpha_matrix = inputs[3]  # alphas
#         Px_ideal = inputs[4]  # ideal (no error) probabilities
#         # C = _tf.reshape(self.dense_correction(_tf.reshape(circuit_encoding, [1, -1])), [-1])
#         epsilon_matrix = self.dense_layer(circuit_encoding)  # depth * num_tracked_error
#         Px_approximate = self.probability_approximation_layer([epsilon_matrix, P, S, scaled_alpha_matrix, Px_ideal]) #+C

#         return Px_approximate# / _tf.reduce_sum(Px_approximate)

#     # @_tf.function
#     def call(self, inputs):
#         return _tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=_tf.float32)

# @_keras.utils.register_keras_serializable(package='Blah1')
# class GraphCircuitErrorVec(_keras.Model):
#     def __init__(self, num_qubits: int, encoding_length: int, modelled_error_generators: list, 
#                  layer_snipper, layer_snipper_args: list,
#                  dense_units=[30, 20, 10, 5, 5], **kwargs):
#         """
#         num_qubits: int
#             The number of qubits that this neural network models.

#         encoding_length: int
#             The number of gate channels in the tensor encoding of the circuits whose fidelity this network
#             predicts.

#         modelled_error_generators: list
#             The primitive error generators that this neural network internally models.

#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

        
#         dense_units: list
#         """
#         super().__init__()
#         self.num_qubits = num_qubits
#         self.modelled_error_generators = _copy.deepcopy(modelled_error_generators)
#         self.number_of_modelled_error_generators = len(self.modelled_error_generators)
#         self.encoding_length = encoding_length
#         self.dense_units = dense_units
#         self.layer_snipper = layer_snipper
#         self.layer_snipper_args = layer_snipper_args

#     def get_config(self):
#         config = super(CircuitErrorVec, self).get_config()
#         config.update({
#             'num_qubits': self.num_qubits,
#             'modelled_error_generators': self.modelled_error_generators,
#             'encoding_length': self.encoding_length,
#             'dense_units': self.dense_units,
#             'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
#             'layer_snipper_args': self.layer_snipper_args,
#         })
#         return config

#     def build(self):
#         self.local_dense = GraphToErrVec(self.layer_snipper, self.layer_snipper_args, self.modelled_error_generators, self.dense_units)
    
#     # @_tf.function
#     def calc_end_of_circ_error_rates(self, M, P, S, scaled_alpha_matrix):
#         """
#         A function that maps the error rates (M) to an end-of-circuit error generator
#         using the permutation matrix P.
#         """
#         signed_M = _tf.math.multiply(S, M) 
#         flat_signed_M, flat_P = _tf.reshape(signed_M, [-1]), _tf.reshape(P, [-1])
#         unique_P, idx = _tf.unique(flat_P) # unique_P values [0, num_error_generators]
#         num_segments = _tf.reduce_max(idx)+1
#         error_rates = _tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
#         gathered_alpha = _tf.gather(scaled_alpha_matrix, unique_P, axis=1)
#         first_order_correction = gathered_alpha*error_rates
#         return first_order_correction

#     # @_tf.function
#     def circuit_to_probability(self, inputs): # will replace this with circ to probability dist
#         """
#         A function that maps a single circuit to the prediction of a 1st order approximate probability vector for each of 2^Q bitstrings.
#         """  
        
#         circuit_encoding = inputs[0] # circuit
#         S = _tf.cast(inputs[1], _tf.float32) # sign matrix
#         P = _tf.cast(inputs[2], _tf.int32) # permutation matrix
#         scaled_alpha_matrix = inputs[3] # alphas (shape is number of tracked error gens, typically 132. Very sparse, could use sparse LA at a later time)
#         Px_ideal = inputs[4] # ideal (no error) probabilities
        
#         epsilon_matrix = self.local_dense(circuit_encoding) # depth * num_tracked_error
#         first_order_correction = self.calc_end_of_circ_error_rates(epsilon_matrix, P, S, scaled_alpha_matrix)
#         Px_approximate = _tf.reduce_sum(first_order_correction, 1) + Px_ideal
#         # Px_approximate_clipped = _tf.reshape(_tf.clip_by_value(Px_approximate, 0, 1), [32])
#         return Px_approximate

#     # @_tf.function
#     def call(self, inputs):
#         output = _tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=_tf.float32)
#         return output
#         # print(output.shape, output.dtype, type(self.dense))
#         # return self.dense(output)

# ------------------------------------------------------------------- #
#        Main part of the QPANNs (input circuit --> error rates matrix)
# ------------------------------------------------------------------- #

# @_keras.utils.register_keras_serializable()
class CircuitToErrorRatesEinSum(_keras.layers.Layer):
    def __init__(self, snipper, modelled_error_generators, dense_units=[30, 20, 10, 5, 5], **kwargs):
        """
        layer_snipper: func
            A function that takes a primitive error generator and maps it to a list that encodes which parts
            of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
            of that primitive error generator.

        modelled_error_generators: list
            A list of the primitive error generators that are to be predicted.
        """
        super().__init__()
        
        self.number_of_modelled_error_generators = len(modelled_error_generators) # This is the output dimension of the network
        self.modelled_error_generators = modelled_error_generators  
        self.stochastic_mask = []
        self.snipper = snipper
        self.dense_units = dense_units + [1] # The + [1] is the output layer.

    def get_config(self):
        config = super().get_config()
        config.update({
            'number_of_modelled_error_generators': self.number_of_modelled_error_generators,
            'modelled_error_generators': self.modelled_error_generators,
            'dense_units': self.dense_units,
            'layer_snipper': self.layer_snipper
        })
        return config
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, input_shape[0], self.number_of_modelled_error_generators)

    # @classmethod
    # def from_config(cls, config):
    #     layer_snipper_config = config.pop("layer_snipper")
    #     layer_snipper = _keras.utils.deserialize_keras_object(layer_snipper_config)
    #     return cls(layer_snipper, **config)
    
    def build(self, input_shape):
        self.dense = EinsumSubNetwork(self.dense_units, self.snipper)        
        super().build(input_shape)
    
    def call(self, inputs):
        max_len_gate_encoding = max([len(layer_encoding) for layer_encoding in self.snipper])
        indices_tensor = _tf.ragged.constant(self.snipper).to_tensor(default_value=-1, 
            shape=[len(self.snipper), max_len_gate_encoding]) # If fewer gate encodings than encoding_length, pad with -1 (illegal index)
        
        # Expand dimensions to match the batch size
        batch_size = _tf.shape(inputs)[0]
        indices_tiled = _tf.tile(_tf.expand_dims(indices_tensor, 0), [batch_size, 1, 1])

        # Create a mask based on the padding (-1 in indices_tensor), so that outputs from these indices can be masked out
        mask = _tf.not_equal(indices_tiled, -1)
        mask = _tf.cast(mask, dtype=inputs.dtype)

        # Change -1 to 0 in indices_tiled before using _tf.gather
        indices_tiled = _tf.where(indices_tiled == -1, _tf.zeros_like(indices_tiled), indices_tiled) # replace indices of -1 (error) to 0 (will point to the wrong index)
        # Gather the values based on the indices
        gathered_slices = _tf.gather(inputs, indices_tiled, batch_dims=1)

        # Apply the mask to zero out the gathered slices at the padding positions
        gathered_slices_masked = gathered_slices * mask

        # print('gathered_slices_masked', gathered_slices_masked.shape)

        # Reshape the gathered slices to concatenate along the last axis
        gathered_slices_flat = _tf.reshape(gathered_slices_masked, [batch_size, self.number_of_modelled_error_generators, -1])

        # print('gathered_slices_flat', gathered_slices_flat.shape)

        # Dense network to learn error rates
        x = _tf.reshape(self.dense(gathered_slices_masked), [-1, self.number_of_modelled_error_generators])

        # A function for squaring a row
        def square(row):
            return row ** 2

        # Expand the mask to match the tensor's shape for broadcasting
        mask_expanded = _tf.expand_dims(_tf.expand_dims(self.stochastic_mask, axis=0), axis=0)

        # Apply the function conditionally using _tf.where
        # If mask_expanded is True, apply custom_function, otherwise keep original row
        x = _tf.where(mask_expanded, square(x), x)

        return x

# ------------------------------------------------------------- #
#        Output layers for the QPANNs (error matrices --> output)
# ------------------------------------------------------------- #
class ProbabilitiesLayer(_keras.layers.Layer):
    """
    TODO
    """
    def __init__(self, **kwargs):
        super(ProbabilitiesLayer, self).__init__(**kwargs)
        self.bitstring_shape = None
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs):
        # TODO : Comment this
        error_rates, P, S, scaled_alpha_matrix, Px_ideal = inputs
        self.bitstring_shape = Px_ideal.shape[0]
        signed_error_rates = _tf.math.multiply(S, error_rates) 
        flat_signed_error_rates, flat_P = _tf.reshape(signed_error_rates, [-1]), _tf.reshape(P, [-1])
        unique_P, idx = _tf.unique(flat_P)  # unique_P values [0, num_error_generators]
        num_segments = _tf.reduce_max(idx) + 1
        summed_error_rates = _tf.math.unsorted_segment_sum(flat_signed_error_rates, idx, num_segments)
        gathered_alpha = _tf.gather(scaled_alpha_matrix, unique_P, axis=1)
        first_order_correction = gathered_alpha * summed_error_rates
        Px_approximate = _tf.reduce_sum(first_order_correction, 1) + Px_ideal
        return Px_approximate

class ProbabilitiesLayerConcise(_keras.layers.Layer):
    """
    TODO
    """
    def __init__(self, **kwargs):
        super(ProbabilitiesLayerConcise, self).__init__(**kwargs)
        self.bitstring_shape = None
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs):
        error_rates, corrections_coefficients, probabilities_ideal = inputs
        # Here we multiple each of the correction coefficients by the corresponding error rate.
        # The first axis of corrections_coefficients is the bit-string axis, so the error_rates
        # tensor is auto-broadcasted across that axis. We then sum up over all but the first axis,
        # computing the summed up effect of all the different errors
        perturbation = _tf.reduce_sum(_tf.math.multiply(corrections_coefficients, error_rates), [1, 2])
        probabilities = probabilities_ideal + perturbation
        return probabilities

class FidelityLayer(_keras.layers.Layer):
    """
    TODO
    """
    def __init__(self, **kwargs):
        super(FidelityLayer, self).__init__(**kwargs)
        self.bitstring_shape = None
    
    def compute_output_shape(self, input_shape):
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs):
        error_rates, P, S = inputs
        self.bitstring_shape = Px_ideal.shape[0]
        signed_error_rates = _tf.math.multiply(S, error_rates) 
        flat_signed_error_rates, flat_P = _tf.reshape(signed_error_rates, [-1]), _tf.reshape(P, [-1])
        unique_P, idx = _tf.unique(flat_P)  # unique_P values [0, num_error_generators]
        num_segments = _tf.reduce_max(idx) + 1
        return None
        
    def calc_masked_err_rates(error_rates, P, mask):
        masked_error_rates = _tf.math.multiply(_tf.cast(mask, _tf.float32), error_rates)
        masked_P = _tf.math.multiply(mask, P)
        flat_masked_error_rates, flat_masked_P = _tf.reshape(masked_error_rates, [-1]), _tf.reshape(masked_P, [-1])
        unique_masked_P, idx = _tf.unique(flat_masked_P)
        num_segments = _tf.reduce_max(idx) + 1
        return _tf.math.unsorted_segment_sum(flat_masked_error_rates, idx, num_segments)
    
    def call(self, inputs):
        """
        A function that maps the error rates (error_rates) to an end-of-circuit error generator
        using the permutation matrix P.
        """
        
        try:
            error_rates, P, S, stochastic_mask, hamiltonian_mask = inputs
        except:
            'Incorrectly formatted inputs. Should be (error_rates, P, S, stochastic_mask, hamiltonian_mask)'


        signed_error_rates = _tf.math.multiply(S, error_rates)
        final_stochastic_error_rates = calc_masked_err_rates(signed_error_rates, P, stochastic_mask)
        final_hamiltonian_error_rates = calc_masked_err_rates(signed_error_rates, P, hamiltonian_mask)
        return _tf.reduce_sum(final_stochastic_error_rates) + _tf.reduce_sum(_tf.square(final_hamiltonian_error_rates))

# TIM COMMENTED OUT THIS CODE AS IT WAS NOT UPDATED IN HIS GENERAL CODE RE-WRITE.
# @_keras.utils.register_keras_serializable(package='Blah1')
# class GraphCircuitErrorVecLookback(_keras.Model):
#     def __init__(self, num_qubits: int, encoding_length: int, modelled_error_generators: list, 
#                  layer_snipper, layer_snipper_args: list,
#                  dense_units=[30, 20, 10, 5, 5], **kwargs):
#         """
#         num_qubits: int
#             The number of qubits that this neural network models.

#         encoding_length: int
#             The number of gate channels in the tensor encoding of the circuits whose fidelity this network
#             predicts.

#         modelled_error_generators: list
#             The primitive error generators that this neural network internally models.

#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

        
#         dense_units: list
#         """
#         super().__init__()
#         self.num_qubits = num_qubits
#         self.modelled_error_generators = _copy.deepcopy(modelled_error_generators)
#         self.number_of_modelled_error_generators = len(self.modelled_error_generators)
#         self.encoding_length = encoding_length
#         self.dense_units = dense_units
#         self.layer_snipper = layer_snipper
#         self.layer_snipper_args = layer_snipper_args

#     def get_config(self):
#         config = super(CircuitErrorVec, self).get_config()
#         config.update({
#             'num_qubits': self.num_qubits,
#             'modelled_error_generators': self.modelled_error_generators,
#             'encoding_length': self.encoding_length,
#             'dense_units': self.dense_units,
#             'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
#             'layer_snipper_args': self.layer_snipper_args,
#         })
#         return config

#     def build(self):
#         self.local_dense = GraphToErrVecLookback(self.layer_snipper, self.layer_snipper_args, self.modelled_error_generators, self.dense_units)
    
#     # @_tf.function
#     def calc_end_of_circ_error_rates(self, M, P, S, scaled_alpha_matrix):
#         """
#         A function that maps the error rates (M) to an end-of-circuit error generator
#         using the permutation matrix P.
#         """
#         signed_M = _tf.math.multiply(S, M) 
#         flat_signed_M, flat_P = _tf.reshape(signed_M, [-1]), _tf.reshape(P, [-1])
#         unique_P, idx = _tf.unique(flat_P) # unique_P values [0, num_error_generators]
#         num_segments = _tf.reduce_max(idx)+1
#         error_rates = _tf.math.unsorted_segment_sum(flat_signed_M, idx, num_segments)
#         gathered_alpha = _tf.gather(scaled_alpha_matrix, unique_P, axis=1)
#         first_order_correction = gathered_alpha*error_rates
#         return first_order_correction

#     # @_tf.function
#     def circuit_to_probability(self, inputs): # will replace this with circ to probability dist
#         """
#         A function that maps a single circuit to the prediction of a 1st order approximate probability vector for each of 2^Q bitstrings.
#         """  
        
#         circuit_encoding = inputs[0] # circuit
#         S = _tf.cast(inputs[1], _tf.float32) # sign matrix
#         P = _tf.cast(inputs[2], _tf.int32) # permutation matrix
#         scaled_alpha_matrix = inputs[3] # alphas (shape is number of tracked error gens, typically 132. Very sparse, could use sparse LA at a later time)
#         Px_ideal = inputs[4] # ideal (no error) probabilities
        
#         epsilon_matrix = self.local_dense(circuit_encoding) # depth * num_tracked_error
#         first_order_correction = self.calc_end_of_circ_error_rates(epsilon_matrix, P, S, scaled_alpha_matrix)
#         Px_approximate = _tf.reduce_sum(first_order_correction, 1) + Px_ideal
#         # Px_approximate_clipped = _tf.reshape(_tf.clip_by_value(Px_approximate, 0, 1), [32])
#         return Px_approximate

#     # @_tf.function
#     def call(self, inputs):
#         output = _tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=_tf.float32)
#         return output
#         # print(output.shape, output.dtype, type(self.dense))
#         # return self.dense(output)

        
# class GraphToErrVecLookback(_keras.layers.Layer):
#     def __init__(self, layer_snipper, layer_snipper_args, modelled_error_generators, dense_units = [30, 20, 10, 5, 5], **kwargs):
#         """
#         layer_snipper: func
#             A function that takes a primitive error generator and maps it to a list that encodes which parts
#             of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
#             of that primitive error generator.

#         modelled_error_generators: list
#             A list of the primitive error generators that are to be predicted.
#         """
#         super().__init__()
        
#         self.number_of_modelled_error_generators = len(modelled_error_generators)  # This is the output dimenision of the network
#         self.modelled_error_generators = modelled_error_generators  
#         self.layer_encoding_indices_for_error_gen = [layer_snipper(error_gen, *layer_snipper_args) for error_gen in modelled_error_generators]
#         self.dense_units = dense_units + [1]
#         self.layer_snipper = layer_snipper
#         self.layer_snipper_args = layer_snipper_args

#     def get_config(self):
#         config = super(LocalizedDenseToErrVec, self).get_config()
#         config.update({
#             'modelled_error_generators': self.modelled_error_generators,
#             'dense_units': self.dense_units,
#             'layer_snipper': _keras.utils.serialize_keras_object(self.layer_snipper),
#             'layer_snipper_args': self.layer_snipper_args
#         })
#         return config
    
#     def compute_output_shape(self, input_shape):
#         # Define the output shape based on the input shape and the number of tracked error generators
#         return (None, input_shape[0], self.number_of_modelled_error_generators)

#     @classmethod
#     def from_config(cls, config):
#         layer_snipper_config = config.pop("layer_snipper")
#         layer_snipper = _keras.utils.deserialize_keras_object(layer_snipper_config)
#         return cls(layer_snipper, **config)
    
#     def build(self, input_shape):
#         # self.dense = {error_gen_idx: DenseSubNetwork(1, self.dense_units) for error_gen_idx in range(self.number_of_modelled_error_generators)}
#         self.dense = [DenseSubNetwork(self.dense_units) for error_gen_idx in range(self.number_of_modelled_error_generators)]
#         super().build(input_shape)

#     def call(self, inputs):
#         x = []
#         zero_layer = _tf.zeros_like(_tf.gather(inputs, self.layer_encoding_indices_for_error_gen[0], axis=-1))
#         for i in range(self.number_of_modelled_error_generators):
#             curr_layers = _tf.gather(inputs, self.layer_encoding_indices_for_error_gen[i], axis=-1)
#             prev_layers = _tf.gather(_tf.roll(inputs, shift=1, axis=-2), self.layer_encoding_indices_for_error_gen[i], axis=-1)
#             # Mask to remove the last circuit layer that has been rolled to the start
#             mask = _tf.concat([_tf.zeros_like(prev_layers[..., 0:1, :]), _tf.ones_like(prev_layers[..., 1:, :])], axis=-2)
#             # Multiply the rolled tensor by the mask to set the layer that corresponds to the "-1 layer" of circuit to zero
#             prev_layers = prev_layers * mask
#             input_pair = _tf.concat([curr_layers, prev_layers], axis=-1)
 
#             x.append(self.dense[i](input_pair))
#         x = _tf.concat(x, axis=-1)
 
#         #indices_tensor = self.layer_encoding_indices_for_error_gen
#         #x = [self.dense[i](_tf.gather(inputs, self.layer_encoding_indices_for_error_gen[i], axis=-1)) 
#         #     for i in range(0, self.number_of_modelled_error_generators)]
#         #x = _tf.concat(x, axis=-1)
#         return x
        