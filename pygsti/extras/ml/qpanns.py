"""Quantum physics aware neural networks (QPANNs).

This module defines Keras models and layers implementing the QPANN architecture described
in https://arxiv.org/abs/2406.05636 (and subsequent internal evolution).

At a high level, a QPANN:
  1) Encodes a circuit as a tensor (depth x encoding_length).
  2) Predicts per-layer error rates for a set of modelled elementary error generators.
  3) Combines predicted error rates with precomputed circuit-specific coefficients to
     produce first-order (or, with `probability_computation='second-order'`, second-order)
     approximations to outcome probabilities (or other metrics). The perturbative
     approximation implemented here is that of "Efficient simulation of Clifford circuits
     with small Markovian errors" (arXiv:2504.15128).
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
from pygsti.extras.ml import customlayers as _cl


@_keras.utils.register_keras_serializable(package='pygsti.extras.ml')
class QPANN(_keras.Model):
    """
    A quantum-physics-aware neural network (QPANN), which is a neural network model for a noisy quantum
    computer. QPANNs are based on the neural network structure introduced in https://arxiv.org/abs/2406.05636,
    but have developed since that paper in various ways.

    The model maps circuit encodings (plus auxiliary propagation/alpha information) to
    predicted outcome probabilities using a first-order error model.

    Attributes
    ----------
    encoding_length : int
        Per-layer encoding feature length.
    modelled_error_generators : list
        List of elementary error generators this model predicts rates for.
    snipper : list[list[int]]
        Feature-selection specification: for each error generator, which encoding indices
        are relevant when predicting its rate.
    probability_computation : {'concise','expanded','second-order'}
        Chooses between the probability-computation layers: 'concise' and 'expanded' are two
        implementations of the first-order approximation; 'second-order' additionally includes
        the correction quadratic in the error rates (see `ProbabilitiesLayerSecondOrder`).
    """

    def __init__(self, encoding_length : int, modelled_error_generators: list,  snipper: list,
                 dense_units: list[int] = [30, 20, 10, 5, 5], probability_computation: str = 'concise', **kwargs) -> None:
        """
        Initialize a QPANN, with random weights and biases. An initialized QPANN should be trained
        on data, to create meaningful predictions and internal model parameters. The inputs for a
        QPANN are circuits, represented as tensors, along with additional information about how
        errors combined in those circuits and impact circuit outcomes. See the docstring of `call`
        for more information on the inputs to a QPANN.

        Parameters
        ----------
        encoding_length : int
            The length of the vectors used to encode each layer of a circuit. If a CircuitEncoder
            object was used to create the encoding, this is CircuitEncoder.length

        modelled_error_generators : list
            A list of the "elementary error generators" that the QPANN models. Each element of this
            list is a tuple. The first element of the tuple is a string specifying the error
            generator type: 'H' or 'S', for Hamiltonian and stochastic errors (currently active
            and Pauli-correlation errors are not supported by QPANNs). The second element of
            the tuple is a single-element tuple where that single element is a string for the
            Pauli indexing the error (e.g., for 4 qubits, this could be 'XYZI').

        snipper : list[list[int]]
             For each error generator, a list of encoding indices to extract as inputs
            to its rate-prediction subnetwork.

        dense_units : list[int], default [30, 20, 10, 5, 5]
            A list of integers, specifying the number of dense units in the neural network
            that learns the mapping from a circuit to matrix of rates of each error in that
            this QPANN models.

        probability_computation : {'concise','expanded','second-order'}, default 'concise'
            Selects the probability approximation implementation. 'concise' and 'expanded' both
            implement the first-order (in the error rates) approximation; 'second-order' also
            includes the term quadratic in the error rates (the second-order correction of
            arXiv:2504.15128), and requires the additional inputs computed by
            `pygsti.extras.ml.encoding.error_generator_tensors` with `order=2`.


        Returns
        -------
        QPANN
            A QPANN, initialized with random weights and biases and ready to be trained on data.

        """
        super().__init__()
        self.encoding_length = encoding_length
        self.modelled_error_generators = _copy.deepcopy(modelled_error_generators)
        self.snipper = _copy.deepcopy(snipper)
        self.dense_units = dense_units.copy()
        self.probability_computation = probability_computation
        # Masks that find the 'S' (stochastic) and 'H' (Hamiltonian) error generators.
        # NOTE: these are plain numpy arrays, not tf.constant(...). A tf.constant created here
        # (eagerly, at __init__ time) gets captured by reference into whatever FuncGraph happens
        # to be active at construction time. Keras 3's Model.fit wraps train_step in nested
        # tf.functions, and if this layer's build()/call() is first triggered from inside that
        # trace, TF raises `InaccessibleTensorError: ... is out of scope` when it later tries to
        # use the stale tf.constant from the (by-then-closed) construction-time graph. Plain
        # numpy/Python values don't have this problem: TF converts them fresh in whatever graph
        # context actually needs them.
        self.stochastic_mask = _np.array([i[0] == 'S' for i in self.modelled_error_generators])

    def get_config(self) -> dict:
        """Return a serializable config dictionary for Keras model saving/loading."""
        # This is required for any custom Keras model. Just contains all of the custom attributes.
        # Must all be serializable or have a way specified to serialize them.
        config = super().get_config()
        config.update({
            'encoding_length': self.encoding_length,
            'modelled_error_generators': self.modelled_error_generators,
            'snipper': self.snipper,
            'dense_units': self.dense_units,
            'probability_computation': self.probability_computation,
            'stochastic_mask': self.stochastic_mask,
        })
        return config

    def build(self) -> None:
        """Instantiate sublayers used by this QPANN (rate predictor + probability layer)."""
        self.dense_layer = CircuitToErrorRatesEinSum(self.snipper, self.modelled_error_generators, self.dense_units)
        if self.probability_computation == 'expanded':
            self.probability_approximation_layer = ProbabilitiesLayer()
        elif self.probability_computation == 'concise':
            self.probability_approximation_layer = ProbabilitiesLayerConcise()
        elif self.probability_computation == 'second-order':
            self.probability_approximation_layer = ProbabilitiesLayerSecondOrder()

    def circuit_to_probability(self, inputs: list | tuple) -> _tf.Tensor:
        """
        Core function that predicts a circuit's output probabilities from the inputs, which include a tensor
        representation of the circuit and other information about the circuits. This function is called by
        `call`.

        Parameters
        ----------
        inputs : list
            Per-circuit inputs. Expected structure depends on `self.probability_computation`:

            * If 'expanded':
                [circuit_encoding, signs, permutations, scaled_alpha_matrix, probabilities_ideal]
            * If 'concise':
                [circuit_encoding, corrections_coefficients, probabilities_ideal]
            * If 'second-order':
                [circuit_encoding, corrections_coefficients, signs, positions,
                 pair_coefficients, probabilities_ideal]

        Returns
        -------
        tf.Tensor
            Probability vector over bitstrings (shape `(2**n,)`).
        """
        circuit_encoding = inputs[0]  # circuit
        # Computes the error rates matrix, which has shape (circuit depth , self.modelled_error_generators)
        error_rates = self.dense_layer(circuit_encoding)
        # The "expanded" original form of the probability computation. It is slower than the 'concise'
        # version, as more computation is done within the network, but it is more amenable to future
        # changes (e.g., a second-order approximation).
        if self.probability_computation == 'expanded':
            signs = _tf.cast(inputs[1], _tf.float32)  # sign matrix
            permutations = _tf.cast(inputs[2], _tf.int32)  # permutation matrix
            scaled_alpha_matrix = inputs[3]  # alpha coefficients, in a 2**n by 2 * 4**n array
            probabilities_ideal = inputs[4]  # ideal (no error) probabilities
            probabilities = self.probability_approximation_layer([error_rates, permutations, signs, scaled_alpha_matrix, probabilities_ideal])

        elif self.probability_computation == 'concise':
            corrections_coefficients = inputs[1]  # The alpha coefficients in a (circuit depth, self.modelled_error_generators) array
            probabilities_ideal = inputs[2]  # ideal (no error) probabilities
            probabilities = self.probability_approximation_layer([error_rates, corrections_coefficients, probabilities_ideal])

        elif self.probability_computation == 'second-order':
            corrections_coefficients = inputs[1]  # first-order alpha coefficients, (2**n, circuit depth, self.modelled_error_generators)
            signs = _tf.cast(inputs[2], _tf.float32)  # propagation sign matrix, (circuit depth, self.modelled_error_generators)
            # positions of each slot's end-of-circuit error generator in this circuit's unique index list.
            # Keras casts all model inputs to floats, so cast back to int for indexing (the values are small
            # integers, exactly representable in float32).
            positions = _tf.cast(inputs[3], _tf.int32)
            pair_coefficients = inputs[4]  # second-order correction coefficients, (2**n, num unique, num unique)
            probabilities_ideal = inputs[5]  # ideal (no error) probabilities
            probabilities = self.probability_approximation_layer(
                [error_rates, corrections_coefficients, signs, positions, pair_coefficients, probabilities_ideal])
        else:
            raise ValueError("Invalid probability_computation choice: " + str(self.probability_computation))

        return probabilities

    def call(self, inputs: list | tuple | _tf.Tensor) -> _tf.Tensor:
        """Vectorize `circuit_to_probability` over a batch using `tf.map_fn`."""
        return _tf.map_fn(self.circuit_to_probability, inputs, fn_output_signature=_tf.float32)


# ------------------------------------------------------------------- #
#        Main part of the QPANNs (input circuit --> error rates matrix)
# ------------------------------------------------------------------- #

class CircuitToErrorRatesEinSum(_keras.layers.Layer):
    """Layer mapping a circuit encoding to per-error-generator error rates.

    This layer uses a "snipper" (list of index lists) to gather relevant features
    for each modelled error generator, then applies an `EinsumSubNetwork` to predict
    rates.

    Output is a tensor of shape `(depth, num_modelled_error_generators)` for each circuit.
    """

    def __init__(self, snipper: list[list[int]], modelled_error_generators: list, dense_units: list[int] = [30, 20, 10, 5, 5], **kwargs) -> None:
        """
        # layer_snipper: func
        #     A function that takes a primitive error generator and maps it to a list that encodes which parts
        #     of a circuit layer to `snip out` as input to dense neural network that predicts the error rate
        #     of that primitive error generator.

        Initialize the layer.

        Parameters
        ----------
        snipper : list[list[int]]
            For each modelled error generator, the indices to gather from the circuit encoding.
        modelled_error_generators : list
            List of primitive error generators whose rates are  to be predicted.
        dense_units : list[int], default [30,20,10,5,5]
            Hidden layer widths; a final output unit is appended internally.

        """
        super().__init__()

        self.number_of_modelled_error_generators = len(modelled_error_generators) # This is the output dimension of the network
        self.modelled_error_generators = modelled_error_generators
        self.snipper = snipper
        self.dense_units = dense_units + [1] # The + [1] is the output layer.
        # Mask that finds the 'S' (stochastic) error generators, so their rates can be squared
        # to enforce non-negativity (see EinsumSubNetwork's caller below).
        # NOTE: a plain numpy array, not tf.constant(...) -- see the identical note in
        # QPANN.__init__ for why (avoids `InaccessibleTensorError` during `.fit()` on Keras 3).
        self.stochastic_mask = _np.array([i[0] == 'S' for i in self.modelled_error_generators])

    def get_config(self) -> dict:
        """Return a serializable config dictionary for Keras layer saving/loading."""

        config = super().get_config()
        config.update({
            'number_of_modelled_error_generators': self.number_of_modelled_error_generators,
            'modelled_error_generators': self.modelled_error_generators,
            'dense_units': self.dense_units,
            'layer_snipper': self.snipper,
            'stochastic_mask': self.stochastic_mask,
        })
        return config

    def compute_output_shape(self, input_shape: tuple | list) -> tuple:
        """Compute output shape: `(None, depth, num_modelled_error_generators)`."""
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, input_shape[0], self.number_of_modelled_error_generators)

    def build(self, input_shape: tuple | list) -> None:
        """Instantiate the internal subnetwork used to predict per-generator rates."""
        self.dense = EinsumSubNetwork(self.dense_units, self.snipper)
        super().build(input_shape)

    def call(self, inputs: _tf.Tensor) -> _tf.Tensor:
        """Predict per-layer error rates from a circuit encoding.

        Parameters
        ----------
        inputs : tf.Tensor
            Circuit encoding for one circuit, shape `(depth, encoding_length)` (or batch-compatible
            shape where leading dimension is batch when called in batch mode).

        Returns
        -------
        tf.Tensor
            Error rates tensor of shape `(batch, num_modelled_error_generators)` as currently implemented.
        """
        max_len_gate_encoding = max([len(layer_encoding) for layer_encoding in self.snipper])
        indices_tensor = _tf.ragged.constant(self.snipper).to_tensor(default_value=-1,
            shape=[len(self.snipper), max_len_gate_encoding]) # If fewer gate encodings than encoding_length, pad with -1 (illegal index)

        # Expand dimensions to match the batch size
        batch_size = _tf.shape(inputs)[0]
        indices_tiled = _tf.tile(_tf.expand_dims(indices_tensor, 0), _tf.stack([batch_size, 1, 1]))

        # Create a mask based on the padding (-1 in indices_tensor), so that outputs from these indices can be masked out
        mask = _tf.not_equal(indices_tiled, -1)
        mask = _tf.cast(mask, dtype=inputs.dtype)

        # Change -1 to 0 in indices_tiled before using _tf.gather
        indices_tiled = _tf.where(indices_tiled == -1, _tf.zeros_like(indices_tiled), indices_tiled) # replace indices of -1 (error) to 0 (will point to the wrong index)
        # Gather the values based on the indices
        gathered_slices = _tf.gather(inputs, indices_tiled, batch_dims=1)

        # Apply the mask to zero out the gathered slices at the padding positions
        gathered_slices_masked = gathered_slices * mask

        # Reshape the gathered slices to concatenate along the last axis
        gathered_slices_flat = _tf.reshape(gathered_slices_masked, [batch_size, self.number_of_modelled_error_generators, -1])

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


class EinsumSubNetwork(_keras.layers.Layer):
    """Subnetwork used by `CircuitToErrorRatesEinSum` to predict rates per error generator.

    Internally this is a `keras.Sequential` made of `CustomDense` layers whose parameters
    are replicated over a leading "error generator" dimension, enabling vectorized
    per-generator predictions.
    """

    def __init__(self, units: list[int], snipper: list[list[int]]) -> None:
        """Initialize the subnetwork.

        Parameters
        ----------
        units : list[int]
            Layer widths; last element is output dimension per generator.
        snipper : list[list[int]]
            Snipper specification; used here to determine number of modelled error generators.
        """
        super().__init__()
        self.units = units
        self.outdim = units[-1]
        self.number_of_modelled_error_generators = len(snipper)
        self.snipper = snipper

    def build(self, input_shape: tuple | list) -> None:
        """Build the underlying Sequential model (stack of CustomDense layers)."""
        init = _keras.initializers.RandomUniform(minval=-0.0001, maxval=0.0001)

        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential(
            [_cl.CustomDense(i, self.number_of_modelled_error_generators, activation='linear') for i in self.units[:-1]] +
            [_cl.CustomDense(self.units[-1], self.number_of_modelled_error_generators, activation='linear', kernel_initializer=init, bias_initializer=init)])

    def get_config(self) -> dict:
        """Return serializable config for Keras."""
        config = super().get_config()
        config.update({
            'outdim': self.outdim,
            'units': self.units
        })
        return config

    def call(self, inputs: _tf.Tensor) -> _tf.Tensor:
        """Forward pass through the subnetwork."""
        return self.sequential(inputs)


# ------------------------------------------------------------- #
#        Output layers for the QPANNs (error matrices --> output)
# ------------------------------------------------------------- #

class ProbabilitiesLayer(_keras.layers.Layer):
    """Expanded probability-approximation layer.

    This layer consumes:
      * predicted per-layer error rates
      * propagation permutation indices `P`
      * propagation signs `S`
      * a dense scaled alpha matrix
      * ideal probabilities

    and outputs first-order corrected probabilities.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the layer."""
        super(ProbabilitiesLayer, self).__init__(**kwargs)
        self.bitstring_shape = None

    def compute_output_shape(self, input_shape):
        """Return output shape `(None, bitstring_shape)` once bitstring_shape is known."""
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs: list | tuple) -> _tf.Tensor:
        """Compute first-order corrected probabilities using dense alpha representation.

        Parameters
        ----------
        inputs : list
            [error_rates, P, S, scaled_alpha_matrix, Px_ideal]

        Returns
        -------
        tf.Tensor
            Approximate probability vector.
        """
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
    """Concise probability-approximation layer used by current QPANN workflow.

    This layer expects correction coefficients already aligned with each circuit's
    error-rate tensor, enabling a simple elementwise multiply-and-sum.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the layer."""
        super(ProbabilitiesLayerConcise, self).__init__(**kwargs)
        self.bitstring_shape = None

    def compute_output_shape(self, input_shape: tuple | list) -> tuple:
        """Return output shape `(None, bitstring_shape)` once bitstring_shape is known."""
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs: list | tuple) -> _tf.Tensor:
        """Compute first-order corrected probabilities using concise coefficients.

        Parameters
        ----------
        inputs : list
            [error_rates, corrections_coefficients, probabilities_ideal]

        Returns
        -------
        tf.Tensor
            Approximate probability vector.
        """
        error_rates, corrections_coefficients, probabilities_ideal = inputs
        # Here we multiple each of the correction coefficients by the corresponding error rate.
        # The first axis of corrections_coefficients is the bit-string axis, so the error_rates
        # tensor is auto-broadcasted across that axis. We then sum up over all but the first axis,
        # computing the summed up effect of all the different errors
        perturbation = _tf.reduce_sum(_tf.math.multiply(corrections_coefficients, error_rates), [1, 2])
        probabilities = probabilities_ideal + perturbation
        return probabilities


class ProbabilitiesLayerSecondOrder(_keras.layers.Layer):
    r"""Second-order probability-approximation layer.

    Implements the outcome probability approximation of "Efficient simulation of Clifford
    circuits with small Markovian errors" (arXiv:2504.15128) to second order in the error
    generator rates. After propagation, the noisy circuit is
    `exp(L'_D) ... exp(L'_1) U_ideal` where `L'_l` is layer `l`'s error generator propagated to
    the end of the circuit (larger `l` = later in the circuit = applied after). Expanding that
    product of exponentials to exact degree 2 in the rates:

        p(bs) = p_ideal(bs)
              + sum_l <L'_l>                                    (first order, as in 'concise')
              + sum_{l2 > l1} <L'_{l2} L'_{l1}> + (1/2) sum_l <L'_l L'_l>   (second order)

    where `<.>` denotes the (scaled) first-order sensitivity (alpha) of bitstring `bs` to an
    end-of-circuit error generator. At degree 2 this is equivalent to the paper's second-order
    BCH recombination plus second-order Taylor correction. Each `L'_l` is the rate-weighted sum
    of the modelled error generators' propagated (signed) images, so the second-order term is a
    quadratic form in the predicted rates whose (rate-independent) coefficients -- the
    sensitivities of `bs` to the pairwise *compositions* of the circuit's unique end-of-circuit
    error generators -- are precomputed by
    `pygsti.extras.ml.encoding.second_order_outcome_correction_tensors`.

    This layer evaluates that quadratic form efficiently: it scatters the signed predicted
    rates into "unique end-of-circuit error generator" space (via `positions`), forms the
    ordered-pair rate products with an exclusive cumulative sum over layers (which reproduces
    exactly the `l2 > l1` ordering and the same-layer factor of 1/2 above), and contracts them
    with the precomputed pair coefficients.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the layer."""
        super(ProbabilitiesLayerSecondOrder, self).__init__(**kwargs)
        self.bitstring_shape = None

    def compute_output_shape(self, input_shape: tuple | list) -> tuple:
        """Return output shape `(None, bitstring_shape)` once bitstring_shape is known."""
        # Define the output shape based on the input shape and the number of tracked error generators
        return (None, self.bitstring_shape)

    def call(self, inputs: list | tuple) -> _tf.Tensor:
        """Compute second-order corrected probabilities.

        Parameters
        ----------
        inputs : list
            [error_rates, corrections_coefficients, signs, positions, pair_coefficients,
             probabilities_ideal], where (with D = circuit depth, E = number of modelled error
            generators, U = number of unique end-of-circuit error generators, n = num qubits):

            * error_rates : (D, E) predicted rates.
            * corrections_coefficients : (2**n, D, E) first-order (sign-weighted) alpha
              coefficients, exactly as in the 'concise' layer.
            * signs : (D, E) propagation signs (0 for padding layers, which removes them from
              the second-order term; the first-order coefficients are already zero there).
            * positions : (D, E) int positions of each slot's propagated end-of-circuit error
              generator in the circuit's unique index list.
            * pair_coefficients : (2**n, U, U) second-order correction coefficients;
              `[bs, u, v]` is the sensitivity of `bs` to the composition of unique generator
              `u` applied AFTER unique generator `v`.
            * probabilities_ideal : (2**n,) ideal probabilities.

        Returns
        -------
        tf.Tensor
            Approximate probability vector, shape `(2**n,)`.
        """
        error_rates, corrections_coefficients, signs, positions, pair_coefficients, probabilities_ideal = inputs
        self.bitstring_shape = probabilities_ideal.shape[0]

        # CircuitToErrorRatesEinSum emits its (D, E) rates with a leading broadcast dimension
        # of 1 (an artifact of its stochastic-rate-squaring tf.where); the 'concise' layer
        # absorbs that via broadcasting, but the einsums below need the bare (D, E) shape.
        error_rates = _tf.reshape(error_rates, _tf.shape(signs))

        # First-order term: identical to ProbabilitiesLayerConcise.
        first_order = _tf.reduce_sum(_tf.math.multiply(corrections_coefficients, error_rates), [1, 2])

        # Scatter the signed rates into unique-end-of-circuit-generator space:
        # r[l, u] = sum_j signs[l, j] * error_rates[l, j] * [positions[l, j] == u].
        num_unique = _tf.shape(pair_coefficients)[-1]
        signed_rates = _tf.math.multiply(signs, error_rates)
        slot_one_hot = _tf.one_hot(positions, depth=num_unique, dtype=signed_rates.dtype)
        rates_by_unique = _tf.einsum('de,deu->du', signed_rates, slot_one_hot)

        # Ordered-pair rate products. earlier[l, v] = sum_{l' < l} r[l', v], so
        # T[u, v] = sum_l r[l, u] * (earlier[l, v] + 0.5 * r[l, v])
        #         = sum_{l2 > l1} r[l2, u] r[l1, v] + 0.5 * sum_l r[l, u] r[l, v],
        # exactly the coefficient with which the composition (u after v) enters the
        # second-order term of the product-of-exponentials expansion.
        earlier = _tf.cumsum(rates_by_unique, axis=0, exclusive=True)
        pair_products = _tf.einsum('du,dv->uv', rates_by_unique, earlier + 0.5 * rates_by_unique)

        second_order = _tf.einsum('buv,uv->b', pair_coefficients, pair_products)

        return probabilities_ideal + first_order + second_order
