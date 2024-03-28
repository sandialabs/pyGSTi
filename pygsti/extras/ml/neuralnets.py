import tensorflow as _tf
import keras as _keras
import numpy as _np
from . import tools as qcl


# Hard coding that the number of qubits is 4. TODO: Update this ASAP.
num_qubits = 4
laplace = qcl.laplace(num_qubits)


class EmbedErrorVector(_keras.layers.Layer): # I think this should perform the correct embedding map.
    def __init__(self, embedding_vector, outdim=256):
        super().__init__()
        m = _np.zeros((len(embedding_vector), outdim), float)
        for i, idx in enumerate(embedding_vector):
            m[i, idx] = 1
        self.m = _tf.constant(m, _tf.float32)
        
    def call(self, inputs):
        return _tf.einsum('...i,ij->...j', inputs, self.m)
    
class LocalizedDenseSubNetwork(_keras.layers.Layer):
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
    
class LocalizedDenseToErrVec(_keras.layers.Layer):
    def __init__(self, laplace, hops, error_interactions, num_error_types):
        """
        laplace: the lapalcian matrix for the connectivity of the qubits. It must be 
        a num_qubits X num_qubits numpy.array.
        
        hops: int
        
        outdim: the dimension of the output error vector. This does *not* need to be
        4^num_qubits.
        """
        super().__init__()
        self.error_interactions = error_interactions
        self.num_error_types = num_error_types
        self.outdim = _np.sum(num_error_types)
        self.num_qubits = laplace.shape[0]
        self.hops = hops
        self.laplace = laplace
        
        laplace_power = _np.linalg.matrix_power(laplace, hops)
        nodes_within_hops = []
        for i in range(self.num_qubits):
            nodes_within_hops.append(_np.arange(self.num_qubits)[abs(laplace_power[i, :]) > 0])

        # Used for deciding which parts of the data to take as input
        self.nodes_within_hops = nodes_within_hops 
            
        self.indices_for_error = []
        for i, qubits in enumerate(error_interactions):
            for _ in range(num_error_types[i]):
                # This hard codes that the qubits are numbered from 0.
                relevant_qubits = _np.concatenate([[p for p in self.nodes_within_hops[q]] for q in qubits])
                indices_for_error = _np.concatenate([[6*q+0,6*q+1,6*q+2,6*q+3,6*q+4,6*q+5] for q in relevant_qubits])
                self.indices_for_error.append(indices_for_error)

    def build(self, input_shape):
        self.dense = {}
        for node in range(self.outdim):
            self.dense[node] = LocalizedDenseSubNetwork(1)
        super().build(input_shape)

    def call(self, inputs):
        x = [self.dense[i](_tf.gather(inputs, self.indices_for_error[i], axis=-1)) for i in range(0, self.outdim)]
        x = _tf.concat(x, axis=-1)
        return x
    
class CircuitErrorVec(_keras.Model):
    def __init__(self, laplace, output_dim, subunit, hops, nonzero_rates, input_shape = None):
        super().__init__()
        self.laplace = laplace
        self.hops = hops
        self.output_dim = output_dim
        self.nonzero_rates = nonzero_rates
        self.input_layer = _keras.layers.InputLayer(input_shape=input_shape)
        
        embedder = []
        error_interactions = []
        num_error_types = []
        for key, item in nonzero_rates.items():
            error_interactions.append(key)
            num_error_types.append(len(item))
            embedder += item
        
        self.embedder = embedder
        self.embedding_layer = EmbedErrorVector(self.embedder, 256)

        self.local_dense = subunit(laplace, hops, error_interactions, num_error_types)
        # This mixes up the outputs of the networks for the different elements of the error vector
        #self.output_layer_int = keras.layers.Dense(self.output_dim, activation = 'linear')
        #self.output_layer_final = keras.layers.Dense(self.output_dim, activation = 'linear')
    
        def new_call(self, inputs):
        # This is very slow when it is called on a large number of circuits. It's because it is not implemented as efficiently (map_fn is the slow part)
        # But that may not be an issue if you keep the batch sizes smallish
        def calc_end_of_circ_err_vec(M, P):
            flat_M, flat_P = _tf.reshape(M, [-1]), _tf.reshape(P, [-1])
            num_segments = _tf.reduce_max(flat_P) + 1
            return _tf.math.unsorted_segment_sum(flat_M, flat_P, num_segments)

        def calc_fidelity(final_evec):
            return _tf.reduce_sum(final_evec**2, axis = -1)

        def circuit_to_fidelity(input):
            # This doesn't work correctly because it doesn't use S correctly. 
            # This approach assumes that you can apply the permutation step to the signed error rates
            # But right now S works only after permutating (I think)
            # Regardless, we can test for SPEED
            # We need S to tell us if the i-th entry in the initial vector contributes positively or negatively
            # right now S tells us if the i-th entry in the propogated vector contributes positively or negatively
            C = input[:, 0:24]
            P = _tf.cast(input[:, 24:24+256], _tf.int32)
            S = input[:, 24+256:25+512]
            evecs = self.embedding_layer(self.local_dense(self.input_layer(C)))
            signed_evecs = _tf.math.multiply(S, evecs)
            total_evec = calc_end_of_circ_err_vec(signed_evecs, P)
            return calc_fidelity(total_evec)
        
        return _tf.map_fn(circuit_to_fidelity, inputs)

    def call(self, inputs):
        depth = inputs.shape[-2]      
        cs = inputs[:,:,0:24]
        signs = inputs[:,:,24+256:24+512]        
        # Haven't check how safe this is in terms of accidentally rounding down 1.999999999... to 1.
        permutations = _tf.cast(inputs[:,:,24:24+256], _tf.int32)

        evecs = self.embedding_layer(self.local_dense(self.input_layer(cs)))
        
        #evecs = _tf.transpose(evecs, (0,2,1)) # Is this necessary?????? It is not.
        total_evec = _tf.reduce_sum(_tf.math.multiply(signs, _tf.gather(evecs, permutations, batch_dims=2)), axis=1)
        #return self.output_layer_final(total_evec)
        return _tf.reduce_sum(total_evec**2, axis=-1)
    