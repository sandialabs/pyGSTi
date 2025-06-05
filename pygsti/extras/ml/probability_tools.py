import stim
import numpy as np
from pygsti.circuits import Circuit
from itertools import product
from pygsti.extras.ml.tools import create_error_propagation_matrix, index_to_error_gen, error_gen_to_index


def uniform_support(tableau, return_support=False):
    """
    Note: This function isn't working correctly for some reason. I've sidestepped the problem for now.
    
    Compute the number of bits over which the stabilizer state corresponding to this Tableau
    would have measurement outcomes which are uniformly random.
    
    Parameters
    ----------
    tableau : stim.Tableau
        stim.Tableau corresponding to the stabilizer state we want the uniform support
        for.
    
    return_support : bool, optional (default False)
        If True also returns a list of qubit indices over which the distribution of outcome
        bit strings is uniform.
    """
    #TODO Test for correctness on support
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    num_random = 0
    support = []
    for i in range(len(tableau)):
        z = sim.peek_z(i)
        if z == 0:
            num_random+=1
            support.append(i)
            # For a phase reference, use the smallest state with non-zero amplitude.
        forced_bit = z == -1
        sim.postselect_z(i, desired_value=forced_bit)
    return (num_random, support) if return_support else num_random

def tableau_fidelity(t1: stim.Tableau, t2: stim.Tableau) -> float:
    t3 = t2**-1 * t1
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(t3)

    p = 1
    for q in range(len(t3)):
        e = sim.peek_z(q)
        if e == -1:
            return 0
        if e == 0:
            p *= 0.5
            sim.postselect_z(q, desired_value=False)
    return p

def output_to_tableau(output):
    """
    Map an input output computational basis bit string into a corresponding Tableau which maps the all zero
    state into that state.
    
    Parameters
    ----------
    output : str
        String corresponding to the computational basis state to prepare the Tableau for.
    
    Returns
    -------
    stim.Tableau
        Tableau which maps the all zero string to this computational basis state
    """
    pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in output]))
    #convert this to a stim.Tableau
    pauli_tableau = pauli_string.to_tableau()
    return pauli_tableau

def first_order_probability_correction(errorgen_dict, tableau, desired_bitstring):
    """
    Compute the first-order correction to the probability of the specified bit string.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bit string being measured.
    
    Returns
    -------
    correction : float
        float corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to first order).
    """
    
    num_random = uniform_support(tableau)
    scale = 1/2**(num_random) #TODO: This might overflow
    
    #now get the sum over the alphas and the error generator rate products needed.
    # print(len(errorgen_dict))
    alpha_errgen_prods = [0]*len(errorgen_dict)
    
    for i, (lbl, rate) in enumerate(errorgen_dict.items()):
        alpha_errgen_prods[i] = np.float64(scale*alpha(lbl, tableau, desired_bitstring)*rate)
    
    correction = sum(alpha_errgen_prods)
    return correction.real

def alpha_generator(errorgen_dict, circuit, desired_bitstring, num_qubits):

    if isinstance(circuit, Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit

    num_random = uniform_support(tableau)
    scale = 1/2**(num_random) #TODO: This might overflow
    
    num_errgens = 2*4**(num_qubits)
    scaled_alpha_errgen = np.zeros(num_errgens, dtype=np.float64)
    # scaled_alpha_errgen = []

    for _, (lbl, rate) in enumerate(errorgen_dict.items()):
        index = error_gen_to_index(lbl.errorgen_type, lbl.bel_to_strings())
        # if index + 1 > num_errgens:
        #     raise ValueError(f"Index {index + 1} exceeds the number of error generators ({num_errgens}).") 
        scaled_alpha_errgen[index] = np.float64(scale*alpha(lbl, tableau, desired_bitstring).real)
        # scaled_alpha_errgen.append((scale*alpha(lbl, tableau, desired_bitstring)).real)
    return scaled_alpha_errgen


def rate_generator(errorgen_dict, circuit, num_qubits):
    num_errgens = 2*4**(num_qubits)
    rates = np.zeros(num_errgens, dtype=np.float64)
    # rates = []

    for _, (lbl, rate) in enumerate(errorgen_dict.items()):
        index = error_gen_to_index(lbl.errorgen_type, lbl.bel_to_strings())
        # if index + 1 > num_errgens:
        #     raise ValueError(f"Index {index + 1} exceeds the number of error generators ({num_errgens}).") 
        rates[index] = np.float64(rate)
        # rates.append(rate)
        
    return rates



def pauli_phase_update(pauli, bitstring):
    """
    Takes as input a pauli and a bit string and computes the output bitstring
    and the overall phase that bit string accumulates.
    
    Parameters
    ----------
    pauli : str or stim.PauliString
        Pauli to apply
    
    bitstring : str
        String representing the bit string to apply the pauli to.
    
    Returns
    -------
    Tuple whose first element is the phase accumulated, and whose second element
    is a string corresponding to the updated bit string.
    """
    
    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)
    
    bitstring = [False if bit=='0' else True for bit in bitstring]
    #list of phase correction for each pauli (conditional on 0)
    #Read [I, X, Y, Z]
    pauli_phases_0 = [1, 1, -1j, 1]
    
    #list of the phase correction for each pauli (conditional on 1)
    #Read [I, X, Y, Z]
    pauli_phases_1 = [1, 1, 1j, -1]
    
    #list of bools corresponding to whether each pauli flips the target bit
    pauli_flips = [False, True, True, False]
    
    overall_phase = 1
    indices_to_flip = []
    for i, (elem, bit) in enumerate(zip(pauli, bitstring)):
        if bit:
            overall_phase*=pauli_phases_1[elem]
        else:
            overall_phase*=pauli_phases_0[elem]
        if pauli_flips[elem]:
            indices_to_flip.append(i)
    #if the input pauli had any overall phase associated with it add that back
    #in too.
    overall_phase*=pauli.sign
    #apply the flips to get the output bit string.
    for idx in indices_to_flip:
        bitstring[idx] = not bitstring[idx]
    #turn this back into a string
    output_bitstring = ''.join(['1' if bit else '0' for bit in bitstring])
    
    return overall_phase, output_bitstring

def amplitude_of_state(tableau, desired_state, return_num_random=False):
    
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    n = sim.num_qubits
    
    #convert desired state into a list of bools
    desired_state = [desired_state[i] == '1' for i in range(n)]
    
    # Determine the magnitude of the target state.
    copy = sim.copy()
    num_random = 0
    for q in range(n):
        desired_bit = desired_state[q]
        z = copy.peek_z(q)
        forced_bit = z == -1
        if z == 0:
            num_random += 1
        elif desired_bit != forced_bit: #forced bit is true if the state is |1>, so this is checking whether the bits match.
            if return_num_random:
                return 0, num_random
            else:
                return 0
        copy.postselect_z(q, desired_value=desired_bit)
    #print(f'{num_random=}')
    magnitude = 2**-(num_random / 2)
    # For a phase reference, use the smallest state with non-zero amplitude.
    copy = sim.copy()
    ref_state = [False]*n
    for q in range(n):
        z = copy.peek_z(q)
        forced_bit = z == -1
        ref_state[q] = forced_bit
        copy.postselect_z(q, desired_value=forced_bit)
    if ref_state == desired_state:
        if return_num_random:
            return magnitude, num_random
        else:
            return magnitude

    # Postselect away states that aren't the desired or reference states.
    # Also move the ref state to |00..00> and the desired state to |00..01>.
    copy = sim.copy()
    found_difference = False
    for q in range(n):
        desired_bit =  desired_state[q]
        ref_bit = ref_state[q]
        if desired_bit == ref_bit:
            copy.postselect_z(q, desired_value=ref_bit)
            if desired_bit:
                copy.x(q)
        elif not found_difference:
            found_difference = True
            if q:
                copy.swap(0, q)
            if ref_bit:
                copy.x(0)
        else:
            # Remove difference between target state and ref state at this bit.
            copy.cnot(0, q)
            copy.postselect_z(q, desired_value=ref_bit)

    # The phase difference between |00..00> and |00..01> is what we want.
    # Since other states are gone, this is the bloch vector phase of qubit 0.
    assert found_difference
    s = str(copy.peek_bloch(0))
    
    if s == "+X":
        phase_factor = 1
    if s == "-X":
        phase_factor = -1
    if s == "+Y":
        phase_factor = 1j
    if s == "-Y":
        phase_factor = -1j
    
    if return_num_random:
        return phase_factor*magnitude, num_random
    else:
        return phase_factor*magnitude
    
    raise NotImplementedError("Bug. State isolation failed.")


def phi(tableau, desired_bitstring, P, Q):
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    
    Parameters
    ----------
    tableau : stim.Tableau
        A stim Tableau corresponding to the input stabilizer state.
        
    desired_bitstring : str
        A string of zeros and ones corresponding to the bit string being measured.
        
    P : str or stim.PauliString
        The first pauli string index.
    Q : str or stim.PauliString
        The second pauli string index.
        
    Returns
    -------
    A complex number corresponding to the value of the phi function.
    """
    
    #start by getting the pauli string which maps the all-zeros string to the target bitstring.
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring]))
    
    #map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    #combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string*P
    eff_Q = Q*initial_pauli_string
    
    #now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    #the corresponding phase corrections.
    all_zeros = '0'*len(eff_P)
    phase1, bitstring1 = pauli_phase_update(eff_P, all_zeros)
    phase2, bitstring2 = pauli_phase_update(eff_Q, all_zeros)
    
    #print(f'{phase1=}')
    #print(f'{phase2=}')
    
    #get the amplitude of these two bitstrings in the stabilizer state.
    amp1 = amplitude_of_state(tableau, bitstring1)
    amp2 = amplitude_of_state(tableau, bitstring2) 
    
    #now apply the phase corrections. 
    amp1*=phase1
    amp2*=phase2
    
    #print(f'{amp1=}')
    #print(f'{amp2=}')
    
    #calculate phi.
    #The second amplitude also needs a complex conjugate applied
    phi = amp1*amp2.conjugate()
    
    #phi should ultimately be either 0, +/-1 or +/-i, scaling might overflow
    #so avoid scaling and just identify which of these it should be.
    if abs(phi)>1e-14:
        if abs(phi.real) > 1e-14:
            if phi.real > 0:
                return complex(1)
            else:
                return complex(-1)
        else:
            if phi.imag > 0:
                return 1j
            else:
                return -1j 
    else:
        return complex(0)

def alpha_kyiv(errorgen, tableau, desired_bitstring):
    """
    First-order error generator sensitivity function for probability.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    errgen_type = errorgen[0]
    basis_element_labels = errorgen[1]
    
    if not isinstance(basis_element_labels[0], stim.PauliString):
        basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])
    
    identity_pauli = stim.PauliString('I'*len(basis_element_labels[0]))
    
    if errgen_type == 'H':
        #print(f'{2*phi(tableau, desired_bitstring, basis_element_labels[0], identity_pauli)=}')
        sensitivity = 2*phi(tableau, desired_bitstring, basis_element_labels[0], identity_pauli).imag
        
    elif errgen_type == 'S':
        sensitivity = phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[0]) \
                    - phi(tableau, desired_bitstring, identity_pauli, identity_pauli)
    elif errgen_type == 'C': #TODO simplify this logic
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[1])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli) \
                    + phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli)
        sensitivity =  first_term.real - second_term.real
    else: #A
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[1], basis_element_labels[0])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli) \
                    - phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli)
        sensitivity =  first_term.imag + second_term.imag
    return sensitivity.real
    

def alpha(errorgen, tableau, desired_bitstring):
    """
    First-order error generator sensitivity function for probability.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    errgen_type = errorgen.errorgen_type
    basis_element_labels = errorgen.basis_element_labels
    
    if not isinstance(basis_element_labels[0], stim.PauliString):
        basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])
    
    identity_pauli = stim.PauliString('I'*len(basis_element_labels[0]))
    
    if errgen_type == 'H':
        #print(f'{2*phi(tableau, desired_bitstring, basis_element_labels[0], identity_pauli)=}')
        sensitivity = 2*phi(tableau, desired_bitstring, basis_element_labels[0], identity_pauli).imag
        
    elif errgen_type == 'S':
        sensitivity = phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[0]) \
                    - phi(tableau, desired_bitstring, identity_pauli, identity_pauli)
    elif errgen_type == 'C': #TODO simplify this logic
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[1])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli) \
                    + phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli)
        sensitivity =  first_term.real - second_term.real
    else: #A
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[1], basis_element_labels[0])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli) \
                    - phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli)
        sensitivity =  first_term.imag + second_term.imag
    return sensitivity.real

def stabilizer_probability(tableau, desired_bitstring):
    """
    Calculate the output probability for the specifed output bit string.
    
    TODO: Should be able to do this more efficiently for many bit strings
    by looking at the structure of the uniform support.
    
    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bit string being measured.
    
    Returns
    -------
    p : float
        probability of desired bitstring.
    """
    
    #turn the desired bitstring into a tableau which maps the all zeros state into the desired one.
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring]))
    
    #compute what Gidney calls the tableau fidelity (which in this case gives the probability).
    p = tableau_fidelity(tableau, output_to_tableau(desired_bitstring))
    return p

def approximate_stabilizer_probability(errorgen_dict, circuit, desired_bitstring):
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.
    
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertable to a stim.Tableau.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bit string being measured.
    
    Returns
    -------
    p : float
        Approximate output probability for desired bitstring.
    """
    
    if isinstance(circuit, Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')
        
    ideal_prob = stabilizer_probability(tableau, desired_bitstring)
    first_order_correction = first_order_probability_correction(errorgen_dict, tableau, desired_bitstring)
    return ideal_prob, ideal_prob + first_order_correction


def probabilities_errorgen_prop(error_propagator, target_model, circuit, use_bch=False, bch_order=1, truncation_threshold=1e-14):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                        bch_kwargs={'bch_order':bch_order,
                                                                    'truncation_threshold':truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    ideal_meas = target_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
    #calculate the probabilities.
    prob_vec = np.zeros(len(ideal_meas))
    for i, effect in enumerate(ideal_meas.values()):
        dense_effect = effect.to_dense().copy()
        dense_prep = ideal_prep.to_dense().copy()
        prob_vec[i] = np.linalg.multi_dot([dense_effect.reshape((1,len(dense_effect))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep),1))]).squeeze()
    return prob_vec


def calculate_probability(circuit, bitstring, target_model, errorgen_propagator, num_qubits):
    propagated_errorgen_layer = errorgen_propagator.propagate_errorgens_bch(circuit, bch_order=1)

    # # helper fn to determine non identities

    # for key in [error_gen for error_gen in propagated_errorgen_layer.keys() if error_gen.bel_to_strings()[0].count('I') < (num_qubits - 2)]:
    #     del propagated_errorgen_layer[key]

    
    # rate_values = np.array([rate_[1] for rate_ in propagated_errorgen_layer.items()])
    rate_values = rate_generator(propagated_errorgen_layer, circuit, num_qubits)
    probabilities = np.array([approximate_stabilizer_probability(propagated_errorgen_layer, circuit, bit) for bit in bitstring]).T
    alpha_values = [alpha_generator(propagated_errorgen_layer, circuit, bit, num_qubits) for bit in bitstring]
    ideal_probabilities = probabilities[0]
    approximate_probabilities = probabilities[1]

    return_dict = {'ideal_probabilities':ideal_probabilities,
                  'approximate_probabilities': approximate_probabilities,
                  'alpha_values': alpha_values,
                  'rate_values' : rate_values}
    
    return return_dict


def calculate_probability_kyiv(circuit, bitstring, target_model, errorgen_propagator, num_qubits, tracked_error_gens):
    propagated_errorgen_layer = errorgen_propagator.propagate_errorgens_bch(circuit, bch_order=1)

    # helper fn to determine non identities

    # for key in [error_gen for error_gen in propagated_errorgen_layer.keys() if error_gen.bel_to_strings()[0].count('I') < (num_qubits - 2)]:
    #     del propagated_errorgen_layer[key]

    
    # rate_values = np.array([rate_[1] for rate_ in propagated_errorgen_layer.items()])
    # rate_values = rate_generator(propagated_errorgen_layer, circuit, num_qubits)
    # probabilities = np.array([approximate_stabilizer_probability(propagated_errorgen_layer, circuit, bit) for bit in bitstring]).T
    alpha_values = [alpha_generator(propagated_errorgen_layer, circuit, bit, num_qubits) for bit in bitstring]
    ideal_probabilities = [stabilizer_probability(circuit.convert_to_stim_tableau(), bit) for bit in bitstring] #probabilities[0]
    # approximate_probabilities = probabilities[1]

    return_dict = {'ideal_probabilities':ideal_probabilities,
                  # 'approximate_probabilities': approximate_probabilities,
                  'alpha_values': alpha_values,
                  # 'rate_values' : rate_values
                  }
    
    return return_dict


def alpha_generator_kyiv(circuit, desired_bitstring, num_qubits, tracked_error_gens):

    if isinstance(circuit, Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit

    num_random = uniform_support(tableau)
    scale = 1/2**(num_random) #TODO: This might overflow
    
    num_errgens = 2*4**(num_qubits)
    scaled_alpha_errgen = np.zeros(num_errgens, dtype=np.float64)
    # scaled_alpha_errgen = []

    for lbl in tracked_error_gens:
        index = error_gen_to_index(lbl[0], lbl[1])
        # if index + 1 > num_errgens:
        #     raise ValueError(f"Index {index + 1} exceeds the number of error generators ({num_errgens}).") 
        scaled_alpha_errgen[index] = np.float64(scale*alpha_kyiv(lbl, tableau, desired_bitstring).real)
        # scaled_alpha_errgen.append((scale*alpha(lbl, tableau, desired_bitstring)).real)
    return scaled_alpha_errgen