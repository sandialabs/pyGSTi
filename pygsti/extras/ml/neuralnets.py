
import tensorflow as _tf
import keras as _keras
import numpy as _np
from . import tools as qcl


# Hard coding that the number of qubits is 4. TODO: Update this ASAP.
num_qubits = 4
laplace = qcl.laplace(num_qubits)


class EmbedErrorVector(_keras.Model):

    def __init__(self, embedding_vector, outdim=256):
        super().__init__()
        m = _np.zeros((outdim, len(embedding_vector)), float)
        for i, idx in enumerate(embedding_vector):
            m[idx, i] = 1
        self.m = _tf.constant(m, _tf.float32)
        
    def call(self, inputs):
        return _tf.linalg.matvec(self.m, inputs)


class LocalizedDenseSubNetwork(_keras.Model):

    def __init__(self, outdim):
        super().__init__()
        self.dense1 = _keras.layers.Dense(30, activation='relu')
        self.dense2 = _keras.layers.Dense(20, activation='relu')
        self.dense3 = _keras.layers.Dense(10, activation='relu')
        self.output_layer = _keras.layers.Dense(outdim, kernel_initializer=_keras.initializers.random_uniform(minval=-0.00001, maxval=0.00001))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


class LocalizedDenseToErrVec(_keras.Model):

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
        self.dense = {}
        for node in range(self.outdim):
            self.dense[node] = LocalizedDenseSubNetwork(1)
            
        self.indices_for_error = []
        for i, qubits in enumerate(error_interactions):
            for j in range(num_error_types[i]):
                # This hard codes that the qubits are numbered from 0.
                relevant_qubits = _np.concatenate([[p for p in self.nodes_within_hops[q]] for q in qubits])
                indices_for_error = _np.concatenate([[6*q+0,6*q+1,6*q+2,6*q+3,6*q+4,6*q+5] for q in relevant_qubits])
                self.indices_for_error.append(indices_for_error)

    def call(self, inputs):

        x = [self.dense[i](_tf.gather(inputs, self.indices_for_error[i], axis=-1)) for i in range(0, self.outdim)]
        x = _tf.concat(x, axis=-1)
        return x    



class CircuitErrorVec(_keras.Model):

    def __init__(self, laplace, output_dim, subunit, hops, nonzero_rates):
        super().__init__()
        self.laplace = laplace
        self.hops = hops
        self.output_dim = output_dim
        self.nonzero_rates = nonzero_rates
        
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

    def call(self, inputs):
        depth = inputs.shape[-2]      
        cs = inputs[:,:,0:24]
        signs = inputs[:,:,24+256:24+512]        
        # Haven't check how safe this is in terms of accidentally rounding down 1.999999999... to 1.
        permutations = _tf.cast(inputs[:,:,24:24+256], _tf.int32)
        
        evecs = _tf.stack([self.embedding_layer(self.local_dense(cs[:,i])) for i in range(depth)], axis=-1)
        evecs = _tf.transpose(evecs, (0,2,1))
        total_evec = _tf.reduce_sum(_tf.math.multiply(signs, _tf.gather(evecs, permutations, batch_dims=2)), axis=1)
        #return self.output_layer_final(total_evec)
        return _tf.reduce_sum(total_evec**2, axis=-1)
    