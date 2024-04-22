import tensorflow as _tf
import keras as _keras
import numpy as _np
# from . import tools as qcl
import copy as _copy
import warnings
from pygsti.extras.ml import newtools


class DenseSubNetwork(_keras.layers.Layer):
    def __init__(self, outdim, units = [30, 20, 10, 5, 5]):
        super().__init__()
        self.outdim = outdim
        self.output_layer = _keras.layers.Dense(outdim, kernel_initializer=_keras.initializers.random_uniform(minval=-0.00001, maxval=0.00001))
        self.units = _copy.deepcopy(units)
    def build(self, input_shape):
        # Define the sub-unit's dense layers
        self.sequential = _keras.Sequential([_keras.layers.Dense(i, activation = 'relu') for i in self.units])
        super().build(input_shape)

    def call(self, inputs):
        # This should naturally handle batches....
        x = self.sequential(inputs)
        return self.output_layer(x)

class MeasurementToErrVec(_keras.layers.Layer):
    def __init__(self, tracked_error_gens: list, dense_units: list, input_shape: tuple):
        super(MeasurementToErrVec, self).__init__()

        self.num_tracked_error_gens = len(tracked_error_gens)  # This is the output dimenision of the network
        self.tracked_error_gens = tracked_error_gens  
        self.dense_units = dense_units
    
    def build(self, input_shape):
        self.dense = {error_gen_idx: DenseSubNetwork(1, self.dense_units) for error_gen_idx in range(self.num_tracked_error_gens)}
        super().build(input_shape)

    def call(self, inputs):
        x = [self.dense[i](inputs) for i in range(0, self.num_tracked_error_gens)]
        x = _tf.concat(x, axis=-1)
        return x

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
    

class LocalizedDenseToErrVec(_keras.layers.Layer):
    def __init__(self, layer_snipper, layer_snipper_args, tracked_error_gens, dense_units = [30, 20, 10, 5, 5]):
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
        self.dense_units = dense_units
    
    def build(self, input_shape):
        self.dense = {error_gen_idx: DenseSubNetwork(1, self.dense_units) for error_gen_idx in range(self.num_tracked_error_gens)}
        super().build(input_shape)

    def call(self, inputs):
        x = [self.dense[i](_tf.gather(inputs, self.layer_encoding_indices_for_error_gen[i], axis=-1)) 
             for i in range(0, self.num_tracked_error_gens)]
        x = _tf.concat(x, axis=-1)
        return x

class CircuitErrorVecWithMeasurementsv1(_keras.Model):
    """
        This class defines a NN architecture that takes in: encoded circuits, measurement tensors, and Z-masks.
        The measurement tensor for a circuit has shape (num_qubits,).

        The NN learns to output the infidelity of the measurement layer as well as the infidelity of the circuit.  
    """
    def __init__(self, num_qubits, num_channels, tracked_error_gens, 
                 layer_snipper, layer_snipper_args, 
                 dense_units = [30, 20, 10, 5, 5], input_shapes=[None, None, None]):
        super(CircuitErrorVecWithMeasurementsv1, self).__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = tracked_error_gens
        self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.dense_units = dense_units
        self.input_shapes = input_shapes
        self.circuit_error_vec = CircuitErrorVecScreenZErrors(num_qubits, num_channels, tracked_error_gens, 
                                                 layer_snipper, layer_snipper_args, dense_units, self.input_shapes)
        self.measurement_network = _keras.Sequential([
            _keras.layers.Dense(60, 'relu', input_shape = input_shapes[1]),
            _keras.layers.Dense(30, 'relu'),
            _keras.layers.Dense(1, 'relu')
        ])

    def call(self, inputs):
        circuits, measurements, z_masks  = inputs
        circuit_infidelities = self.circuit_error_vec([circuits, z_masks])
        measurement_infidelities = self.measurement_network(measurements)
        measurement_infidelities = _tf.squeeze(measurement_infidelities, axis=1)
        return circuit_infidelities + measurement_infidelities

class CircuitErrorVecWithMeasurementsAndBitstringsv1(_keras.Model):
    """
        This class defines a NN architecture that takes in: encoded circuits, measurement tensors, and Z-masks.
        The measurement tensor for a circuit has shape (2*num_qubits,). The first num_qubits entries mark the measured qubits,
        while the last num_qubits entries denote a circuit's target bitstring (padded if necessary).

        The NN learns to output the infidelity of the measurement layer as well as the infidelity of the circuit. 

        INPUTS: (circuits, measurements, target_outcomes, z_masks) 
    """
    def __init__(self, num_qubits, num_channels, tracked_error_gens, 
                 layer_snipper, layer_snipper_args, 
                 dense_units = [30, 20, 10, 5, 5], input_shapes=[None, None, None, None]):
        super(CircuitErrorVecWithMeasurementsAndBitstringsv1, self).__init__()
        self.num_qubits = num_qubits
        self.tracked_error_gens = tracked_error_gens
        self.num_tracked_error_gens = len(self.tracked_error_gens)
        self.num_channels = num_channels
        self.len_gate_encoding = self.num_qubits * self.num_channels
        self.dense_units = dense_units
        self.input_shapes = input_shapes
        self.circuit_error_vec = CircuitErrorVecScreenZErrors(num_qubits, num_channels, tracked_error_gens, 
                                                 layer_snipper, layer_snipper_args, dense_units, [self.input_shapes[0], self.input_shapes[-1]])
        self.measurement_network = _keras.Sequential([
            _keras.layers.Dense(60, 'relu'),
            _keras.layers.Dense(30, 'relu'),
            _keras.layers.Dense(1, 'relu')
        ])

    def call(self, inputs):
        circuits, measurements, target_outcomes, z_masks  = inputs
        circuit_infidelities = self.circuit_error_vec([circuits, z_masks])
        combo_measurements_outcomes = _tf.concat([measurements, target_outcomes], axis = 1)
        measurement_infidelities = self.measurement_network(combo_measurements_outcomes)
        measurement_infidelities = _tf.squeeze(measurement_infidelities, axis=1)
        return circuit_infidelities + measurement_infidelities

class CircuitErrorVec(_keras.Model):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, 
                layer_snipper, layer_snipper_args: dict,
                dense_units = [30, 20, 10, 5, 5], input_shape=None):
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
        self.dense_units = dense_units
        self.input_layer = _keras.layers.InputLayer(input_shape=input_shape)
        self.local_dense = LocalizedDenseToErrVec(layer_snipper, layer_snipper_args, self.tracked_error_gens, self.dense_units)
        self.hamiltonian_mask = _tf.constant([1 if error[0] == 'H' else 0 for error in tracked_error_gens], _tf.int32)
        self.stochastic_mask = _tf.constant([1 if error[0] == 'S' else 0 for error in tracked_error_gens], _tf.int32)
    
    def old_call(self, inputs):
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
            unique_P, idx = _tf.unique(flat_P) # This reduces NN performance compared to pre-processing.
            num_segments = _tf.reduce_max(idx) + 1
            error_types = _tf.ones(num_segments) # tensor with a 1 if Hamiltonian, 0 if Stochastic
            return _tf.math.unsorted_segment_sum(flat_M, idx, num_segments), error_types  # This returns a larger vector than necessary

        def calc_fidelity(final_evec, error_types):
            # TO DO: This needs to be generalized when we include 'S' errors.
            # We are going to use error_types to determine if we are going to square or not.
            ham_contribution = _tf.reduce_sum((final_evec*error_types)**2, axis = -1)
            stoch_contribution = _tf.reduce_sum(final_evec*(1-error_types), axis = -1)
            return ham_contribution + stoch_contribution

        def circuit_to_fidelity(input):
            """
            A function that maps a single circuit to the prediction for its process fidelity (a single real number).
            """
            C = input[:, 0:self.len_gate_encoding]
            P = _tf.cast(input[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], _tf.int32)
            S = input[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
            evecs = self.local_dense(self.input_layer(C))
            total_evec, error_types = calc_end_of_circ_err_vec(evecs, P, S)
            return calc_fidelity(total_evec, error_types)
        
        return _tf.map_fn(circuit_to_fidelity, inputs)   

    def call(self, inputs):
        # This is very slow when it is called on a large number of circuits. It's because it is not implemented
        # as efficiently (map_fn is the slow part)
        # But that may not be an issue if you keep the batch sizes smallish

        def calc_masked_err_rates(M, P, mask):
            masked_M = _tf.math.multiply(_tf.cast(mask, _tf.float32), M)
            masked_P = _tf.math.multiply(mask, P)
            flat_masked_M, flat_masked_P = _tf.reshape(masked_M, [-1]), _tf.reshape(masked_P, [-1])
            unique_masked_P, idx = _tf.unique(flat_masked_P)
            num_segments = _tf.reduce_max(idx) + 1
            return _tf.math.unsorted_segment_sum(flat_masked_M, idx, num_segments)
        def calc_end_of_circ_error(M, P, S):
            """
            A function that maps the error rates (M) to an end-of-circuit error generator
            using the permutation matrix P.
            """
            signed_M = _tf.math.multiply(S, M)
            final_stochastic_error_rates = calc_masked_err_rates(signed_M, P, self.stochastic_mask)
            final_hamiltonian_error_rates = calc_masked_err_rates(signed_M, P, self.hamiltonian_mask)
            return _tf.reduce_sum(final_stochastic_error_rates) + _tf.reduce_sum(_tf.square(final_hamiltonian_error_rates))

        def circuit_to_fidelity(input):
            """
            A function that maps a single circuit to the prediction for its process fidelity (a single real number).
            """
            C = input[:, 0:self.len_gate_encoding]
            P = _tf.cast(input[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], _tf.int32)
            S = input[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
            evecs = self.local_dense(self.input_layer(C))

            return calc_end_of_circ_error(evecs, P, S)

        return _tf.map_fn(circuit_to_fidelity, inputs)
            
class CircuitErrorVecScreenZErrors(CircuitErrorVec):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, layer_snipper, layer_snipper_args: dict,
                 dense_units: list, input_shapes=[None, None]):
        super().__init__(num_qubits, num_channels, tracked_error_gens, layer_snipper, layer_snipper_args, dense_units, input_shapes[0])
        self.input_shapes = {'circuit': input_shapes[0], 
                             'z_masks': input_shapes[2]}
    def call(self, inputs):

        def calc_masked_err_rates(M, P, mask):
            masked_M = _tf.math.multiply(_tf.cast(mask, _tf.float32), M)
            masked_P = _tf.math.multiply(mask, P)
            flat_masked_M, flat_masked_P = _tf.reshape(masked_M, [-1]), _tf.reshape(masked_P, [-1])
            unique_masked_P, idx = _tf.unique(flat_masked_P)
            num_segments = _tf.reduce_max(idx) + 1
            return _tf.math.unsorted_segment_sum(flat_masked_M, idx, num_segments)
        
        def calc_end_of_circ_error(M, P, S, z_mask):
            """
            A function that maps the signed error rates (S*M) to an end-of-circuit error generator
            using the permutation matrix P. It also screens out errors that are Z-type on the 
            measured qubits.
            """
            signed_M = _tf.math.multiply(S, M)
            # Zero out the entries in signed_M that get sent to errors that are Z-type Paulis on the active qubits
            masked_signed_M = _tf.math.multiply(signed_M, z_mask)
            final_stochastic_error_rates = calc_masked_err_rates(masked_signed_M, P, self.stochastic_mask)
            final_hamiltonian_error_rates = calc_masked_err_rates(masked_signed_M, P, self.hamiltonian_mask)
            return _tf.reduce_sum(final_stochastic_error_rates) + _tf.reduce_sum(_tf.square(final_hamiltonian_error_rates))

        def circuit_to_fidelity(input):
            """
            A function that maps a single circuit to the prediction for its process fidelity (a single real number).
            """
            circuit, z_mask = input
            circuit, z_mask = _tf.cast(circuit, _tf.float32), _tf.cast(z_mask, _tf.float32)
            # circuit, measurement = _tf.unstack(input)
            C = circuit[:, 0:self.len_gate_encoding]
            P = _tf.cast(circuit[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], _tf.int32)
            S = circuit[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
            evecs = self.local_dense(self.input_layer(C))
            return calc_end_of_circ_error(evecs, P, S, z_mask)
        # TO DO: Put the signed_M and masked_signed_M calculations outside of circuit_to_fidelity
        return _tf.map_fn(circuit_to_fidelity, inputs, fn_output_signature=_tf.float32)

class CircuitErrorVecScreenZErrorsWithMeasurementsBitstrings(CircuitErrorVec):
    def __init__(self, num_qubits: int, num_channels: int, tracked_error_gens: list, tracked_error_indices: list,
                layer_snipper, layer_snipper_args: dict,
                dense_units: list, input_shapes=[None, None, None, None, None]):
        super().__init__(num_qubits, num_channels, tracked_error_gens, layer_snipper, layer_snipper_args, dense_units, input_shapes[0])
        self.input_shapes = {'circuit': input_shapes[0], 'measurements': input_shapes[1], 'target_outcomes': input_shapes[2]
                             'z_masks': input_shapes[3], 'measurement_masks': input_shapes[4]}
        self.meas_bitstring_shape = input_shapes[1][0]+input_shapes[3][0]
        self.measurement_dense = MeasurementToErrVec(self.tracked_error_gens, self.dense_units, tuple(meas_bitstring_shape,))
        self.error_indices = _tf.constant(tracked_error_indices, dtype = _tf.int32)

    def call(self, inputs):

        def calc_masked_err_rates(M, P, mask):
            masked_M = _tf.math.multiply(_tf.cast(mask, _tf.float32), M)
            masked_P = _tf.math.multiply(mask, P)
            flat_masked_M, flat_masked_P = _tf.reshape(masked_M, [-1]), _tf.reshape(masked_P, [-1])
            unique_masked_P, idx = _tf.unique(flat_masked_P)
            num_segments = _tf.reduce_max(idx) + 1
            return _tf.math.unsorted_segment_sum(flat_masked_M, idx, num_segments)
        
        def calc_end_of_circ_error(M, P, S, z_mask):
            """
            A function that maps the signed error rates (S*M) to an end-of-circuit error generator
            using the permutation matrix P. It also screens out errors that are Z-type on the 
            measured qubits.
            """
            signed_M = _tf.math.multiply(S, M)
            # Zero out the entries in signed_M that get sent to errors that are Z-type Paulis on the active qubits
            masked_signed_M = _tf.math.multiply(signed_M, z_mask)
            final_stochastic_error_rates = calc_masked_err_rates(masked_signed_M, P, self.stochastic_mask)
            final_hamiltonian_error_rates = calc_masked_err_rates(masked_signed_M, P, self.hamiltonian_mask)
            return _tf.reduce_sum(final_stochastic_error_rates) + _tf.reduce_sum(_tf.square(final_hamiltonian_error_rates))

        def circuit_to_fidelity(input):
            """
            A function that maps a single circuit to the prediction for its process fidelity (a single real number).
            """
            circuit, measurement, target_outcome, z_mask, measurement_mask  = input
            # circuit, z_mask = _tf.cast(circuit, _tf.float32), _tf.cast(z_mask, _tf.float32)
            
            C = circuit[:, 0:self.len_gate_encoding]
            P = _tf.cast(circuit[:, self.len_gate_encoding:self.len_gate_encoding + self.num_tracked_error_gens], _tf.int32)
            S = circuit[:, self.len_gate_encoding + self.num_tracked_error_gens:self.len_gate_encoding + 2 * self.num_tracked_error_gens]
            combo_measurements_outcomes = _tf.concat([measurement, target_outcome], axis = 1)
            
            circuit_evecs = self.local_dense(self.input_layer(C))
            measurement_evecs = self.measurement_dense(combo_measurements_outcomes)

            all_evecs = _tf.stack([circuit_evecs, measurement_evecs])
            padded_P = _tf.stack([P, self.error_indices])
            padded_S = _tf.stack([S, _tf.ones([1,len(self.num_tracked_error_gens)])])
            padded_z_mask = _tf.stack([z_mask, measurement_mask])

            return calc_end_of_circ_error(all_evecs, padded_P, padded_S, padded_z_mask)
        # TO DO: Put the signed_M and masked_signed_M calculations outside of circuit_to_fidelity
        return _tf.map_fn(circuit_to_fidelity, inputs, fn_output_signature=_tf.float32)