import numpy as _np
import warnings as _warnings
from pygsti.processors.processorspec import QubitProcessorSpec as QPS
from pygsti.extras.ml.tools import create_error_propagation_matrix, index_to_error_gen, error_gen_to_index
from tqdm import trange
import time

###### Functions that encode a circuit into a tensor ###

geometry_cnot_channels = {'ring': 4, 'linear': 4, 'bowtie': 12, 't-bar': 8, 'algiers-t-bar': 8, 'grid': 8, 'melbourne': 8} # you get 2 channels for each cnot gate

def compute_channels(pspec: QPS, geometry: str) -> int:
    return len(pspec.gate_names) - 1 + geometry_cnot_channels[geometry]
           
def clockwise_cnot(g, num_qubits):
    return (g.qubits[0] - g.qubits[1]) % num_qubits == num_qubits - 1

def melbourne_gate_to_index(g, q, pspec)-> int:
    '''
    Works for Melbourne
    '''
    assert(q in g.qubits)
    single_qubit_gates = list(pspec.gate_names)
    single_qubit_gates.remove('Gcnot')
    if g.name == 'Gcnot':
        if g.qubits in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(13,12),(12,11),(11,10),(10,9),(9,8),(8,7)]:
            if q == g.qubits[0]: return 0
            else: return 1
        elif g.qubits in [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(12,13),(11,12),(10,11),(9,10),(8,9),(7,8)]:
            if q == g.qubits[0]: return 2
            else: return 3
        elif g.qubits in [(1,13),(2,12),(3,11),(4,10),(5,9),(6,8)]:
            if q == g.qubits[0]: return 4
            else: return 5
        elif g.qubits in [(13,1),(12,2),(11,3),(10,4),(9,5),(8,6)]:
            if q == g.qubits[0]: return 6
            else: return 7
    elif g.name in pspec.gate_names:
        # We have a single-qubit Clifford gate
        return geometry_cnot_channels['melbourne']+single_qubit_gates.index(g.name) # we put the single-qubit gates after the CNOT channels.
    else:
        raise ValueError('Invalid gate name for this encoding!')

def bowtie_gate_to_index(g, q, pspec)-> int:
    '''
    Works for Yorktown.
    '''
    assert(q in g.qubits)
    single_qubit_gates = list(pspec.gate_names)
    single_qubit_gates.remove('Gcnot')
    if g.name == 'Gcnot':
        if g.qubits in [(0,1), (3,4)]:
            if q == g.qubits[0]: return 0
            else: return 1
        elif g.qubits in [(1,0), (4,3)]:
            if q == g.qubits[0]: return 2
            else: return 3
        elif g.qubits in [(0,2), (2,4)]:
            if q == g.qubits[0]: return 4
            else: return 5
        elif g.qubits in [(2,0), (4,2)]:
            if q == g.qubits[0]: return 6
            else: return 7
        elif g.qubits in [(1,2), (2,3)]:
            if q == g.qubits[0]: return 8
            else: return 9
        elif g.qubits in [(2,1), (3,2)]:
            if q == g.qubits[0]: return 10
            else: return 11
        else:
            raise ValueError('Invalid gate name for this encoding!')
    elif g.name in pspec.gate_names:
        # We have a single-qubit Clifford gate
        return geometry_cnot_channels['bowtie']+single_qubit_gates.index(g.name) # we put the single-qubit gates after the CNOT channels.
    else:
        raise ValueError('Invalid gate name for this encoding!')


def algiers_t_bar_gate_to_index(g, q, pspec)-> int:
    '''
    Works for algiers, etc.
    '''
    assert(q in g.qubits)
    single_qubit_gates = list(pspec.gate_names)
    single_qubit_gates.remove('Gcnot')
    if g.name == 'Gcnot':
        if g.qubits in [(0,1), (1,4)]:
            if q == g.qubits[0]: return 0
            else: return 1
        elif g.qubits in [(1,0), (4,1)]:
            if q == g.qubits[0]: return 2
            else: return 3
        elif g.qubits in [(1,2), (2,3)]:
            if q == g.qubits[0]: return 4
            else: return 5
        elif g.qubits in [(2,1), (3,2)]:
            if q == g.qubits[0]: return 6
            else: return 7
        else:
            raise ValueError('Invalid gate name for this encoding!')
    elif g.name in pspec.gate_names:
        # We have a single-qubit Clifford gate
        return 8+single_qubit_gates.index(g.name) # we put the single-qubit gates after the CNOT channels.
    else:
        raise ValueError('Invalid gate name for this encoding!')

def t_bar_gate_to_index(g, q, pspec)-> int:
    '''
    Works for London, Burlington, Essex, Vigo, Belem
    '''
    assert(q in g.qubits)
    single_qubit_gates = list(pspec.gate_names)
    single_qubit_gates.remove('Gcnot')
    single_qubit_gates.remove('Gecres')
    if g.name == 'Gcnot' or g.name=='Gecres':
        if g.qubits in [(0,1), (1,2)]:
            if q == g.qubits[0]: return 0
            else: return 1
        elif g.qubits in [(1,0), (2,1)]:
            if q == g.qubits[0]: return 2
            else: return 3
        elif g.qubits in [(1,3), (3,4)]:
            if q == g.qubits[0]: return 4
            else: return 5
        elif g.qubits in [(3,1), (4,3)]:
            if q == g.qubits[0]: return 6
            else: return 7
        else:
            raise ValueError('Invalid gate name for this encoding!')
    elif g.name in pspec.gate_names:
        # We have a single-qubit Clifford gate
        return 8+single_qubit_gates.index(g.name) # we put the single-qubit gates after the CNOT channels.
    else:
        raise ValueError('Invalid gate name for this encoding!')

def ring_gate_to_index(g, q, pspec):
    assert(q in g.qubits)
    num_qubits = pspec.num_qubits
    multi_quibit_list = ['Gecr', 'Gecres', 'Gcnot']
    single_qubit_gates = [gate for gate in list(pspec.gate_names) if gate not in multi_quibit_list]
    # single_qubit_gates = list(pspec.gate_names)
    # single_qubit_gates.remove('Gecr')
    if g.name in multi_quibit_list:
        qs = g.qubits
        if q == g.qubits[0] and clockwise_cnot(g, num_qubits):
            return 0
        if q == g.qubits[1] and clockwise_cnot(g, num_qubits):
            return 1
        if q == g.qubits[0] and not clockwise_cnot(g, num_qubits):
            return 2
        if q == g.qubits[1] and not clockwise_cnot(g, num_qubits):
            return 3
        else:
            raise ValueError('Invalid gate name for this encoding!')
    elif g.name in pspec.gate_names:
        # We have a single-qubit gate
        return 4+single_qubit_gates.index(g.name) # we put the single-qubit gates after the CNOT/ECR channels.
    else:
        raise ValueError('Invalid gate name for this encoding!')
    
def layer_to_matrix(layer, num_qubits = None, num_channels = None, 
                    indexmapper = None, indexmapper_kwargs = {}, 
                    valuemapper = None, valuemapper_kwargs = {}) -> _np.array:
    '''
    Function that encodes a layer into a matrix. 
        
    valuemapper: a function that maps a gate to a specific value
    
    indexmapper: a function that maps a gate to an index.
    '''
    if valuemapper is None: valuemapper = lambda x: 1
    if num_qubits is None: num_qubits = layer.num_lines
    assert(num_channels is not None), 'I need to know the number of channels per qubit!'
    assert(indexmapper is not None), 'I need a way to map a gate to an index!!!'

    mat = _np.zeros((num_qubits, num_channels), float)
    for g in layer:
        for q in g.qubits:
            if type(q) == str: q_index = int(q[1:])
            else: q_index = q
            mat[q_index, indexmapper(g, q, **indexmapper_kwargs)-1] = valuemapper(g, **valuemapper_kwargs)
    return mat

def circuit_to_tensor(circ, depth = None, num_qubits = None, num_channels = None, add_measurements = False, 
                      indexmapper = None, indexmapper_kwargs = None, 
                      valuemapper = None, valuemapper_kwargs = {}) -> _np.array:
    '''
    Function that transforms a circuit into a numpy array/tensor.

    valuemapper: a function that maps gates to numeric values.

    indexmapper: a function that maps gates to indices.

    The add measurements functionality assumes that gates are always encoded as a postive value.
    '''
    
    if depth is None: depth = circ.depth
    if num_qubits is None: num_qubits = circ.num_lines
    assert(num_channels is not None), 'I need to know how many channels there are per qubit.'
    ctensor = _np.zeros((num_qubits, depth, num_channels), float)
    for i in range(circ.depth):
        ctensor[:, i, :] = layer_to_matrix(circ.layer(i), num_qubits, num_channels, 
                                           indexmapper, indexmapper_kwargs, 
                                           valuemapper, valuemapper_kwargs)
    if add_measurements:
        row_sums = _np.sum(ctensor, axis = (1, 2)) # Figure out which qubits are dark (i.e., unused)
        used_qubits = _np.where(row_sums != 0)  
        ctensor[used_qubits[0], -1, -1] = 1       
    return ctensor

def active_qubits(ctensor):
    row_sums = _np.sum(ctensor, axis = (1,2))
    used_qubits = _np.where(row_sums != 0)
    measurement = _np.zeros(ctensor.shape[0])
    measurement[used_qubits[0]] = 1 
    return measurement

# def screen_z_errors(P, measurement):
#     """
#     A function that takes in a circuit's permutation matrix and its measurement tensor (i.e., the matrices that tell you where every 
#     error vector gets mapped to at the end of the circuit) and creates a mask that masks out
#     all error's that are Z-type on the active qubits in a circuit. 
#     """
#     active_qubits = _tf.where(measurement == 1)[:, 0]
#     flattened_P = _tf.reshape(P, [-1])
#     unique_P, _ = _tf.unique(flattened_P) # Get the unique values in P as well as their indices. 
#     condition_mask = _tf.map_fn(lambda x: good_error(x, active_qubits), unique_P, fn_output_signature=_tf.bool)
#     good_errors = _tf.boolean_mask(unique_P, condition_mask)
#     expand_flat_P = _tf.expand_dims(flattened_P, axis = -1)
#     masked_P = _tf.reduce_any(_tf.equal(expand_flat_P, good_errors), axis = -1)
#     masked_P = _tf.reshape(masked_P, P.shape)
#     masked_P = _tf.cast(masked_P, _tf.float32)
    
#     return masked_P

def good_error(error_index, measured_qubits, num_qubits):
    typ, pauli = index_to_error_gen(error_index, num_qubits)
    paulistring = _np.array(list(pauli[0]))
    active_paulis = paulistring[measured_qubits]
    return _np.any((active_paulis == 'X') | (active_paulis == 'Y'))
    
def z_mask(P: _np.array, measurement: _np.array):
    """
    A function that takes in a circuit's permutation matrix and its measurement tensor (i.e., the matrices that tell you where every 
    error vector gets mapped to at the end of the circuit) and creates a mask that masks out
    all error's that are Z-type on the active qubits in a circuit. 
    """
    num_qubits = len(measurement)
    measured_qubits = _np.where(measurement == 1)[0]
    good_errors = _np.vectorize(lambda x: good_error(x, measured_qubits, num_qubits))(P)
    return good_errors.astype(float)

def unique_value_mapping(A):
    """
    Finds the unique values in a ND numpy array, orders them by size,
    and creates a mapping from each unique value to its index in the ordering.

    Parameters:
    A (numpy.ndarray): A ND numpy array.

    Returns:
    dict: A dictionary mapping each unique value to its index in the sorted list of unique values.
    """
    # Find the unique values in the array
    unique_values = _np.unique(A)
    
    # Sort the unique values
    sorted_unique_values = _np.sort(unique_values)
    
    # Create a mapping from each unique value to its index in the sorted array
    value_to_index_mapping = {value: index for index, value in enumerate(sorted_unique_values)}
    
    return value_to_index_mapping

def map_array_values(A, mapping):
    """
    Maps the entries of an ND numpy array to their corresponding indices
    based on a provided mapping of unique values to indices.

    Parameters:
    A (numpy.ndarray): A 3D numpy array.
    mapping (dict): A dictionary mapping unique values to their indices.

    Returns:
    numpy.ndarray: A new 3D numpy array with values mapped to their corresponding indices.
    """
    # Vectorize the mapping function
    vectorized_mapping = _np.vectorize(mapping.get)
    
    # Apply the mapping to the array
    mapped_A = vectorized_mapping(A)
    
    return mapped_A
        
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
                    
    for i in trange(len(circs), smoothing=0):
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
    for i in trange(len(circs), smoothing=0):
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


        
        