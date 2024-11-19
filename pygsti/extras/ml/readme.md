Setup precompute for P matrix and indices

```python

len_gate_encoding = num_qubits * num_channels
num_tracked_error_gens = len(error_gens)


def get_idx(input, mask):
    idx_array = np.zeros((input.shape[0], circuits['train'].shape[1]*num_tracked_error_gens), dtype=np.int32)
    P = input[:, :, len_gate_encoding:len_gate_encoding + num_tracked_error_gens]
    for i in range(input.shape[0]):
        masked_P = tf.math.multiply(mask, P[i])
        flat_masked_P = tf.reshape(masked_P, [-1])
        _, idx = tf.unique(flat_masked_P)
        idx_array[i] = idx
    return idx_array

def create_dataset(circuits, infidelities, error_gens, batch_size, precompute=False):
    hamiltonian_mask = [1 if error[0] == 'H' else 0 for error in error_gens]
    stochastic_mask = [1 if error[0] == 'S' else 0 for error in error_gens]
    # Ensure infidelities is reshaped correctly
    infidelities = infidelities.reshape(-1, 1)  # Ensure it's a 2D array for consistency

    # Create a dataset that yields ((circuit, idx_h, idx_s), infidelity)
    if precompute:
        idx_h = get_idx(circuits, hamiltonian_mask)  # (n_data, 23760)
        idx_s = get_idx(circuits, stochastic_mask)    # (n_data, 23760)
        dataset = tf.data.Dataset.from_tensor_slices(((circuits, idx_h, idx_s), infidelities))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((circuits, infidelities))
    
    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch data for better performance
    
    return dataset
```


train_dataset_precompute  = create_dataset(circuits['train'], infidelities['train'], error_gens, batch_size=363, precompute=True)
validate_dataset_precompute = create_dataset(circuits['validate'], infidelities['validate'], error_gens, batch_size=500, precompute=True)

train_dataset  = create_dataset(circuits['train'], infidelities['train'], error_gens, batch_size=363, precompute=False)
validate_dataset = create_dataset(circuits['validate'], infidelities['validate'], error_gens, batch_size=500, precompute=False)


Example run for GPU optimized precompute model

```python

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
circuit_dense = ml.neuralnets_gpu_optimization.CircuitErrorVec(num_qubits, num_channels, error_gens, layer_snipper,
                                                layer_snipper_args, input_shape=(180, 308), dense_units=[30, 10])

circuit_dense.compile(optimizer, 
                      loss=keras.losses.MeanSquaredError(),
                     metrics=['mae'])

for inputs, targets in train_dataset_precompute.take(1):
    dummy_output = circuit_dense(inputs)

circuit_dense.summary()  # Now this should show the parameters

```

Example run for CPU optimized precompute model

```python

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
circuit_dense = ml.neuralnets_cpu_optimization.CircuitErrorVec(num_qubits, num_channels, error_gens, layer_snipper,
                                                layer_snipper_args, input_shape=(180, 308), dense_units=[30, 10])

circuit_dense.compile(optimizer, 
                      loss=keras.losses.MeanSquaredError(),
                     metrics=['mae'])

for inputs, targets in train_dataset.take(1):
    dummy_output = circuit_dense(inputs)

circuit_dense.summary()  # Now this should show the parameters
```



