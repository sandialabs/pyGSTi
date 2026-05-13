"""Custom neural network layers for use in quantum physics aware neural networks (QPANNs).

This module defines TensorFlow/Keras layers used by QPANN models. The layers here support
specialized connectivity patterns and parameterizations commonly needed for quantum-physics-
motivated learning tasks.

Layers
------
SelectiveDense
    Applies multiple dense projections to different selected subsets of the input features.
CustomDense
    A Dense-like layer with parameters replicated over a leading "error generator" dimension.
"""
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
        """Create a dense-like layer that applies separate kernels to selected input subsets.

        This layer takes a single input tensor and, for each entry in `input_indices`,
        gathers the specified input feature indices and applies a dedicated dense
        transformation (kernel) to that subset. The per-subset outputs are stacked
        along a new axis.

        Parameters
        ----------
        units : int
            Number of output units (features) produced by each subset-specific dense
            transformation.
        input_indices : Sequence[Sequence[int]]
            A list (or other sequence) where each element is an iterable of integer
            indices selecting which input features to use for one "group".
            The number of groups is `len(input_indices)`.
        activation : str or callable, optional
            Activation function to apply. Passed through `tf.keras.activations.get`.
        use_bias : bool, default True
            Whether to include and add a bias term.
        kernel_initializer : str or tf.keras.initializers.Initializer, default 'glorot_uniform'
            Initializer for the kernel weights.
        bias_initializer : str or tf.keras.initializers.Initializer, default 'zeros'
            Initializer for the bias vector.
        kernel_regularizer : str or tf.keras.regularizers.Regularizer, optional
            Regularizer for the kernel weights.
        bias_regularizer : str or tf.keras.regularizers.Regularizer, optional
            Regularizer for the bias vector.
        kernel_constraint : str or tf.keras.constraints.Constraint, optional
            Constraint applied to the kernel weights.
        bias_constraint : str or tf.keras.constraints.Constraint, optional
            Constraint applied to the bias vector.
        **kwargs
            Additional keyword arguments passed to the base `Layer`.

        Notes
        -----
        If the input has shape `(batch, input_dim)` and there are `G = len(input_indices)`
        groups, the output shape is `(batch, G, units)`.

        A single bias vector of shape `(units,)` is shared across all groups.
        """
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
        """Create the layer weights based on the input shape.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor. The last dimension is interpreted as the
            input feature dimension.

        Creates
        -------
        self.kernels : list[tf.Variable]
            One kernel per group, with shape `(len(indices), units)`.
        self.bias : tf.Variable or None
            Shared bias vector of shape `(units,)` if `use_bias` is True.
        """
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
        """Apply the selective dense transformations.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape `(batch, input_dim)` (or any tensor where the last
            axis corresponds to the feature dimension referenced by `input_indices`).

        Returns
        -------
        tf.Tensor
            Output tensor of shape `(batch, num_groups, units)`, where `num_groups`
            is `len(input_indices)`.

        Notes
        -----
        For each group `g`, the layer computes:

        \[
        y_g = X[:, indices_g] W_g + b
        \]

        where \(W_g\) has shape \((|indices_g|, units)\) and \(b\) is shared.
        """
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
        
        """Create a Dense-like layer with parameters replicated over error generators.

        This layer generalizes `tf.keras.layers.Dense` by introducing a leading
        dimension `num_errorgens` on the kernel (and bias). It is intended for
        settings where a separate linear map is maintained for each "error generator"
        (or similar model index).

        Parameters
        ----------
        units : int 
            Dimensionality of the output space (number of output features).
        num_errorgens : int
            Number of error-generator slices. The kernel will have shape
            `(num_errorgens, input_dim, units)` and the bias will have shape
            `(num_errorgens, units)` if enabled.
        activation : str or callable, optional
            Activation function to apply. Passed through Keras' activation handling.
        use_bias : bool, default True
            Whether to include and add a bias term.
        kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer,
        kernel_constraint, bias_constraint :
            Standard Keras `Dense` configuration options.
        **kwargs
            Additional keyword arguments passed to the base `Dense`/`Layer`.
        """

        super(CustomDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                            kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint, **kwargs)
        self.num_errorgens = num_errorgens

    def build(self, input_shape):
        """Create the layer weights based on the input shape.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input tensor. The last dimension is interpreted as the
            input feature dimension.

        Creates
        -------
        self._kernel : tf.Variable
            Kernel tensor of shape `(num_errorgens, input_dim, units)`.
        self.bias : tf.Variable or None
            Bias tensor of shape `(num_errorgens, units)` if `use_bias` is True.
        """
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
        """Apply the per-error-generator dense transformations.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor expected to have shape `(num_errorgens, batch, input_dim)`.
            The leading dimension corresponds to the error-generator index.

        Returns
        -------
        tf.Tensor
            Output tensor of shape `(num_errorgens, batch, units)`.

        Notes
        -----
        The computation is performed via Einstein summation:

        \[
        y[d,f,o] = \sum_m x[d,f,m] \, W[d,m,o]
        \]

        where \(d\) indexes the error generator, \(f\) indexes the batch/examples,
        \(m\) indexes input features, and \(o\) indexes output units.
        """

        x = tf.einsum('dfm,fmo->dfo', inputs, self.kernel)
        if self.bias is not None:
            x = tf.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
