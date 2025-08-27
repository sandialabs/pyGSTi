def create_input_data(circs:list, fidelities:list, tracked_error_gens: list, 
                      pspec, geometry: str, num_qubits = None, num_channels = None, 
                      measurement_encoding = None, idealouts = None,
                      indexmapper = None, indexmapper_kwargs = {}, 
                      valuemapper = None, valuemapper_kwargs = {},
                      max_depth = None, return_separate=False, stimDict = None,
                      large_encoding = False):
    '''
    Maps a list of circuits and fidelities to numpy arrays of encoded circuits and fidelities. 

    Args:
       - tracked_error_gens: a list of the tracked error generators.
       - pspec: the processor on which the circuits are defined. Used to determine the number of qubits and channels (optional)
       - geometry: the geometry in which you plan to embed the circuits (i.e., ring, grid, linear). Optional.
       - num_qubits: the number of qubits (optional, if pspec and geometry are specified)
       - num_channels: the number of channels used to embed a (qubit, gate) pair (optional, if pspec and geometry are specified.)
       - indexmapper: function specifying how to map a gate to a channel.
       - valuemapper: function specifying how to encode each gate in pspec (optional, defaults to assigning each gate a value of 1)
       - measurement_encoding: int or NoneType specifying how to encode measurements. 
            - If NoneType, then no measurements are returned.
            - If 1, then measurements are encoded as extra channels in the circuit tensor.
            - If 2, then the measurements are returned separately in a tensor of shape (num_qubits,)
       - list of each circuit's target outcome. Only used when measurement_encoding = 2.
    '''
    num_circs = len(circs)
    num_error_gens = len(tracked_error_gens)
    

    if max_depth is None: max_depth = _np.max([c.depth for c in circs])
    # print(max_depth)
    
    if num_channels is None: num_channels = compute_channels(pspec, geometry)
    encode_measurements = False
    if measurement_encoding == 1:
        _warnings.warn('Measurement encoding scheme 1 is not implemented yet!')
        encode_measurements = True
        num_channels += 1
        max_depth += 1 # adding an additional layer to each circuit for the measurements.
    elif measurement_encoding == 2:
        # Encodes the measurements as a seperate vector.
        measurements = _np.zeros((num_circs, num_qubits))
        x_zmask = _np.zeros((num_circs, max_depth, num_error_gens), int)
    elif measurement_encoding == 3:
        # Encodes the measurements as a seperate vector and returns a measurement mask.
        measurements = _np.zeros((num_circs, num_qubits))
        x_zmask = _np.zeros((num_circs, max_depth, num_error_gens), int)
        x_mmask = _np.zeros((num_circs, num_error_gens), int)
        tracked_error_indices = _np.array([error_gen_to_index(error[0], error[1]) for error in tracked_error_gens])
    
    if num_qubits is None: num_qubits = len(pspec.qubit_labels)
    if valuemapper is None: valuemapper = lambda x: 1
    assert(indexmapper is not None), 'I need a way to map gates to an index!!!!'

    x_circs = _np.zeros((num_circs, num_qubits, max_depth, num_channels), float)
    x_signs = _np.zeros((num_circs, max_depth, num_error_gens), int)
    x_indices = _np.zeros((num_circs, max_depth, num_error_gens), int)
    if type(fidelities) is list: y = _np.array(fidelities)
                    
    for i, c in enumerate(circs):
        if i % 200 == 0:
            print(i, end=',', flush = True)
        x_circs[i, :, :, :] = circuit_to_tensor(c, max_depth, num_qubits, num_channels, encode_measurements,
                                                         indexmapper, indexmapper_kwargs,
                                                         valuemapper, valuemapper_kwargs 
                                                         )              
        c_indices, c_signs = create_error_propagation_matrix(c, tracked_error_gens, stim_dict = stimDict)
        if large_encoding:
            mapping = unique_value_mapping(c_indices)
            c_indices = map_array_values(c_indices, mapping)
        x_indices[i, 0:c.depth, :] = c_indices # deprecated: np.rint(c_indices)
        x_signs[i, 0:c.depth, :] = c_signs # deprecated: np.rint(c_signs)
        if measurement_encoding == 1:
            # This is where update the signs and indices to account for the measurements
            # NOT IMPLEMENTED!!!!!
            x_signs[i, :, -1] = 1 
            x_indices[i, :, -1] = 0 # ??? Need to figure this out ??? Need to take the tracked error gens and map them to their unique id
        elif measurement_encoding == 2:
            measurements[i, :] = active_qubits(x_circs[i, :, :, :])
            measurements[i, ::-1] = measurements[i, :] # flip it and reverse it
            x_zmask[i, 0:c.depth, :] = z_mask(c_indices, measurements[i, :])
        elif measurement_encoding == 3:
            measurements[i, :] = active_qubits(x_circs[i, :, :, :])
            measurements[i, ::-1] = measurements[i, :] # flip it and reverse it
            x_zmask[i, 0:c.depth, :] = z_mask(c_indices, measurements[i, :])  # Update this
            x_mmask[i, :] = z_mask(tracked_error_indices, measurements[i, :]) # Update this

           
    if return_separate:
        return x_circs, x_signs, x_indices, y

    else:
        len_gate_encoding = num_qubits * num_channels
        xc_reshaped = _np.zeros((x_circs.shape[0], x_circs.shape[2], x_circs.shape[1] * x_circs.shape[3]), float)
        for qi in range(num_qubits): 
            for ci in range(num_channels): 
                xc_reshaped[:, :, qi * num_channels + ci] = x_circs[:, qi, :, ci].copy()
            
        x = _np.zeros((x_indices.shape[0], x_indices.shape[1], 2 * num_error_gens + len_gate_encoding), float)
        x[:, :, 0:len_gate_encoding] = xc_reshaped[:, :, :]
        x[:, :, len_gate_encoding:num_error_gens + len_gate_encoding] = x_indices[:, :, :]
        x[:, :, num_error_gens + len_gate_encoding:2 * num_error_gens + len_gate_encoding] = x_signs[:, :, :]
        if measurement_encoding == 2:
            if idealouts is not None:
                target_outcomes = _np.array([list(idealout) for idealout in idealouts], dtype = float)
                return x, y, measurements, target_outcomes, x_zmask
            return x, y, measurements, x_zmask, 
        elif measurement_encoding == 3:
            if idealouts is not None:
                target_outcomes = _np.array([list(idealout) for idealout in idealouts], dtype = float)
                return x, y, measurements, target_outcomes, x_zmask, x_mmask
            return x, y, measurements, x_zmask, x_mmask

        return x, y


def create_probability_data(circs:list,
                            ideal_probabilities:list, 
                            alpha_values:list,
                            approximate_probabilities:list, 
                            # rate_values:list,
                            tracked_error_gens: list, 
                            pspec, geometry: str, true_probabilities=None, 
                            num_qubits = None, num_channels = None, 
                            measurement_encoding = False, idealouts = None,
                            indexmapper = None, indexmapper_kwargs = {}, 
                            valuemapper = None, valuemapper_kwargs = {},
                            max_depth = None, return_separate=False, stimDict = None,
                            large_encoding = False):
    '''
    Maps a list of circuits and fidelities to numpy arrays of encoded circuits and fidelities. 

    Args:
       - tracked_error_gens: a list of the tracked error generators.
       - pspec: the processor on which the circuits are defined. Used to determine the number of qubits and channels (optional)
       - geometry: the geometry in which you plan to embed the circuits (i.e., ring, grid, linear). Optional.
       - num_qubits: the number of qubits (optional, if pspec and geometry are specified)
       - num_channels: the number of channels used to embed a (qubit, gate) pair (optional, if pspec and geometry are specified.)
       - indexmapper: function specifying how to map a gate to a channel.
       - valuemapper: function specifying how to encode each gate in pspec (optional, defaults to assigning each gate a value of 1)
    '''
    num_circs = len(circs)
    num_error_gens = len(tracked_error_gens)
    

    if max_depth is None: max_depth = _np.max([c.depth for c in circs])
    if num_channels is None: num_channels = compute_channels(pspec, geometry)
    if num_qubits is None: num_qubits = len(pspec.qubit_labels)
    if valuemapper is None: valuemapper = lambda x: 1
    assert(indexmapper is not None), 'I need a way to map gates to an index!!!!'

    x_circs = _np.zeros((num_circs, num_qubits, max_depth, num_channels), float)
    x_signs = _np.zeros((num_circs, max_depth, num_error_gens), int)
    x_indices = _np.zeros((num_circs, max_depth, num_error_gens), int)
    if true_probabilities is not None:
        y_true = _np.array(true_probabilities)
    y_approx = _np.array(approximate_probabilities)
                    
    for i in range(len(circs)):
        c = circs[i]
        x_circs[i, :, :, :] = circuit_to_tensor(c, max_depth, num_qubits, num_channels, measurement_encoding,
                                                         indexmapper, indexmapper_kwargs,
                                                         valuemapper, valuemapper_kwargs 
                                                         )              
        c_indices, c_signs = create_error_propagation_matrix(c, tracked_error_gens, stim_dict = stimDict)
        x_indices[i, 0:c.depth, :] = c_indices # deprecated: np.rint(c_indices)
        x_signs[i, 0:c.depth, :] = c_signs # deprecated: np.rint(c_signs)

    x_alpha = _np.array(alpha_values)
    x_px = _np.array(ideal_probabilities)
    xc_reshaped = _np.zeros((x_circs.shape[0], x_circs.shape[2], x_circs.shape[1] * x_circs.shape[3]), float)
    for qi in range(num_qubits): 
        for ci in range(num_channels): 
            xc_reshaped[:, :, qi * num_channels + ci] = x_circs[:, qi, :, ci].copy()

    if true_probabilities is not None:
        return xc_reshaped, x_signs, x_indices, x_alpha, x_px, y_approx, y_true
    else:
        return xc_reshaped, x_signs, x_indices, x_alpha, x_px, y_approx

def create_probability_data_test(circs: list,
                            ideal_probabilities: list, 
                            alpha_values: list,
                            approximate_probabilities: list, 
                            tracked_error_gens: list, 
                            pspec, geometry: str, true_probabilities=None, 
                            num_qubits=None, num_channels=None, 
                            measurement_encoding=None, idealouts=None,
                            indexmapper=None, indexmapper_kwargs={}, 
                            valuemapper=None, valuemapper_kwargs={},
                            max_depth=None, return_separate=False, stimDict=None,
                            large_encoding=False):
    '''
    Maps a list of circuits and fidelities to numpy arrays of encoded circuits and fidelities. 

    Args:
       - tracked_error_gens: a list of the tracked error generators.
       - pspec: the processor on which the circuits are defined. Used to determine the number of qubits and channels (optional)
       - geometry: the geometry in which you plan to embed the circuits (i.e., ring, grid, linear). Optional.
       - num_qubits: the number of qubits (optional, if pspec and geometry are specified)
       - num_channels: the number of channels used to embed a (qubit, gate) pair (optional, if pspec and geometry are specified.)
       - indexmapper: function specifying how to map a gate to a channel.
       - valuemapper: function specifying how to encode each gate in pspec (optional, defaults to assigning each gate a value of 1)
    '''
    num_circs = len(circs)
    num_error_gens = len(tracked_error_gens)

    if max_depth is None:
        max_depth = np.max([c.depth for c in circs])
    if num_channels is None:
        num_channels = compute_channels(pspec, geometry)
    if num_qubits is None:
        num_qubits = len(pspec.qubit_labels)
    if valuemapper is None:
        valuemapper = lambda x: 1
    assert indexmapper is not None, 'I need a way to map gates to an index!!!!'

    x_circs = _np.zeros((num_circs, num_qubits, max_depth, num_channels), float)
    x_signs = _np.zeros((num_circs, max_depth, num_error_gens), int)
    x_indices = _np.zeros((num_circs, max_depth, num_error_gens), int)
    y_true = _np.array(true_probabilities)
    y_approx = _np.array(approximate_probabilities)

    time_circuit_to_tensor = 0
    time_create_error_propagation_matrix = 0
    time_rest = 0

    start_time = time.time()
    for i in range(len(circs)):
        c = circs[i]

        start = time.time()
        x_circs[i, :, :, :] = circuit_to_tensor(c, max_depth, num_qubits, num_channels, measurement_encoding,
                                                indexmapper, indexmapper_kwargs, valuemapper, valuemapper_kwargs)
        time_circuit_to_tensor += time.time() - start

        start = time.time()
        c_indices, c_signs = create_error_propagation_matrix(c, tracked_error_gens, stim_dict=stimDict)
        time_create_error_propagation_matrix += time.time() - start

        x_indices[i, 0:c.depth, :] = c_indices
        x_signs[i, 0:c.depth, :] = c_signs

    x_alpha = _np.array(alpha_values)
    x_px = _np.array(ideal_probabilities)
    xc_reshaped = _np.zeros((x_circs.shape[0], x_circs.shape[2], x_circs.shape[1] * x_circs.shape[3]), float)

    start = time.time()
    for qi in range(num_qubits):
        for ci in range(num_channels):
            xc_reshaped[:, :, qi * num_channels + ci] = x_circs[:, qi, :, ci].copy()
    time_rest += time.time() - start

    total_time = time.time() - start_time

    print(f"Time for circuit_to_tensor: {time_circuit_to_tensor:.6f} seconds")
    print(f"Time for create_error_propagation_matrix: {time_create_error_propagation_matrix:.6f} seconds")
    print(f"Time for the rest of the function: {time_rest:.6f} seconds")
    print(f"Total time: {total_time:.6f} seconds")
    if true_probabilities is not None:
        return xc_reshaped, x_signs, x_indices, x_alpha, x_px, y_approx, y_true
    else:
        return xc_reshaped, x_signs, x_indices, x_alpha, x_px, y_approx
