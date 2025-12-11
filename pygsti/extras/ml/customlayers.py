""" Custom neural network layers for use in quantum physics aware neural networks (QPANNs) """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints


class SelectiveDense(Layer):
    def __init__(self, units, input_indices, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(SelectiveDense, self).__init__(**kwargs)
        self.units = units
        self.input_indices = input_indices
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels = []
        for indices in self.input_indices:
            kernel = self.add_weight(
                shape=(len(indices), self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel'
            )
            self.kernels.append(kernel)
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias'
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        outputs = []
        for i, indices in enumerate(self.input_indices):
            selected_inputs = tf.gather(inputs, indices, axis=-1)
            output = tf.matmul(selected_inputs, self.kernels[i])
            outputs.append(tf.expand_dims(output, 1))
            # print('output', output.shape)

        
        output = tf.concat(outputs, axis=1)
        # print(output.shape, output.shape)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

# # Example usage
# input_indices = [
#     list(range(0, 100, 2)),  # Neuron 1: even indices
#     list(range(1, 100, 2)),  # Neuron 2: odd indices
#     list(range(0, 50))       # Neuron 3: indices 0 to 49
# ]

# inputs = tf.keras.Input(shape=(100,))
# selective_dense_layer = SelectiveDense(units=3, input_indices=input_indices, activation='relu')
# outputs = selective_dense_layer(inputs)

# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.summary()


class CustomDense(Dense):
    def __init__(self, units, num_errorgens, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(CustomDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint, **kwargs)
        self.num_errorgens = num_errorgens

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self._kernel = self.add_weight(
            name="kernel",
            shape=(self.num_errorgens, input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_errorgens, self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        x = tf.einsum('dfm,fmo->dfo', inputs, self.kernel)
        if self.bias is not None:
            x = tf.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
