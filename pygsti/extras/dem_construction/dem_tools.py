import numpy as np
import stim
import itertools

from collections import Counter
from collections import defaultdict
import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel
from . import helper_functions as _tools
from . import sensitivity_analysis as _svs

def pauli_product(P1, P2):
    P3 = P1*P2
    return (P3.sign, P3 / P3.sign)
    
def alpha_pauli(errorgen, sim, pauli):
    """
    First-order error generator sensitivity function for pauli expectations.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.TableauSimulator
        Stim TableauSimulator corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.
    """  
    
    errgen_type = errorgen.errorgen_type
    basis_element_labels = errorgen.basis_element_labels
    
    if not isinstance(basis_element_labels[0], stim.PauliString):
        basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])
    
    identity_pauli = stim.PauliString('I'*len(basis_element_labels[0]))
    
    if errgen_type == 'H':
        pauli_bel_0_comm = eprop.com(pauli, basis_element_labels[0])
        if pauli_bel_0_comm is not None:
            sign = -1j*pauli_bel_0_comm[0]
            expectation  = sim.peek_observable_expectation(pauli_bel_0_comm[1])
            return np.real_if_close(sign*expectation)
        else: 
            return 0 
    elif errgen_type == 'S':
        if pauli.commutes(basis_element_labels[0]):
            return 0
        else:
            expectation  = sim.peek_observable_expectation(pauli)
            return np.real_if_close(-2*expectation)
    elif errgen_type == 'C': 
        A = basis_element_labels[0]
        B = basis_element_labels[1]
        com_AP = A.commutes(pauli)
        com_BP = B.commutes(pauli) # TODO: can skip computing this in some cases for minor performance boost.
        if A.commutes(B):
            if com_AP:
                return 0
            else:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(-4*expectation)
        else: # {A,B} = 0
            if com_AP:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(-2*expectation)
            else:
                if com_BP:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(2*expectation)
                else:
                    return 0
    else: # A
        A = basis_element_labels[0]
        B = basis_element_labels[1]
        com_AP = A.commutes(pauli)
        com_BP = B.commutes(pauli) # TODO: can skip computing this in some cases for minor performance boost.
        if A.commutes(B):
            if com_AP:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(1j*2*expectation)
            else:
                if com_BP:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(-1j*2*expectation)
                else:
                    return 0
        else: # {A,B} = 0
            if com_AP:
                return 0
            else:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return np.real_if_close(1j*4*expectation)

def up_to_kbits(n, k):
    result = []
    for j in range(k+1):
        for bits in itertools.combinations(range(n), j):
            s = ['0'] * n
            for bit in bits:
                s[bit] = '1'
            result.append(''.join(s))
    return result

def xor_binary_strings(binary_string1, binary_string2):
    """
    Performs XOR operation on two binary strings.

    Args:
        binary_string1: The first binary string.
        binary_string2: The second binary string.

    Returns:
        The XOR result as a binary string.
    """

    if len(binary_string1) != len(binary_string2):
        raise ValueError("Binary strings must be of equal length")

    num1 = int(binary_string1, 2)
    num2 = int(binary_string2, 2)
    xor_result = num1 ^ num2
    return bin(xor_result)[2:].zfill(len(binary_string1))


def format_dem_stim(dem_dict, n_logical):
    '''
    formats a dictionary of event probabilities into a stim dem
    '''
    n_detectors = len(list(dem_dict.keys())[0])
    #detectors called D0,D1,...D(n_detectors-1)
    #need to add in logical detectors as a separate thing
    dem_string = ''
    for dbs, prob in dem_dict.items():
        #print(len(dbs))
        if prob > 0:
            line = f'error({prob}) '+''.join([f'D{i} ' if dbs[i]=='1' else '' for i in range(n_detectors-n_logical)]+[f'L{i} ' if dbs[n_detectors-n_logical+i]=='1' else '' for i in range(n_logical)])+'\n'
            #if dbs[n_detectors-n_logical]=='1':
            #    print('logical')
            dem_string+=line
    return dem_string

def compute_dem_pols_from_z_exps(z_exps, k_max, n_aux, n_rounds, ancilla_qubits):
    '''
    exhaustively compute z polarizations of weight < k_max and put into a dictionary
    n_aux is the number of aux qubits (in one round of syndrome extraction)
    Used primarily in the statevector simulation workflow, but no reason it cannot be used elsewhere
    '''
    dem_pols = {}
    event_bitstrings = up_to_kbits((n_rounds)*n_aux, k_max) #note: these are currently strings
    for bs in event_bitstrings: 
        #bs is a DEM string. We need to translate it into an effective syndrome bit string
        #TODO
        #CHANGED 05-26 because I think it was wrong?
        #bs_eff = bs[:n_aux]+ ''.join(xor_binary_strings(bs[n_aux*i: n_aux*(i+1)], bs[n_aux*(i+1): n_aux*(i+2)]) for i in range(0,n_rounds-1)) #+bs[-n_aux:]
        bs_eff = ''.join(xor_binary_strings(bs[n_aux*i: n_aux*(i+1)], bs[n_aux*(i+1): n_aux*(i+2)]) for i in range(0,n_rounds-1))+bs[-n_aux:]
        
        
        qubit_subset = [q for i, q in enumerate(ancilla_qubits) if bs_eff[i]=='1']
        pauli_eff_ancilla = ''.join(['Z' if bs_eff[i]=='1' else 'I' for i in range(len(bs_eff))])
        pauli_eff = "".join(i + j for i, j in itertools.zip_longest('I'*(n_aux+1), pauli_eff_ancilla, fillvalue=''))

        #print(bs, bs_eff, qubit_subset)
        #print(bs, bs_eff, z_exps[pauli_eff])

        dem_pols[bs] = z_exps[pauli_eff]

    return dem_pols
    
def compute_deltas(eeg_terms, qubit_labels, inverse_tableau, ancilla_qubits):
    #iterate through pairs of propagated Paulis
    #ancilla_qubits: list of expanded circuit aux qubits
    error_pairs = list(itertools.product(eeg_terms, times=2))
  
    deltas = {}

    for eeg1, eeg2 in error_pairs: 
        P = eeg1.basis_element_labels[0]
        Q = eeg2.basis_element_labels[0]
        # Only terms where [P,Q] = 0 contribute
        if P.commutes(Q):
            support = [y for y in list(set(get_pauli_xy_support(P)+get_pauli_xy_support(Q))) if y in ancilla_qubits]
            #allowed_bitstrings = itertools.product(['0','1'], times=len(support)) #bitstrings on all subsets of the ancilla support 

            ys_to_include = itertools.powerset(support)#identify ancilla qubits included in P, P'
            #STILL NEED BIT STRING Y
            
            for subset_with_z in ys_to_include:
                # The measurement Pauli operator

                #turn y into a subset of ancilla_qubits
                #subset_with_z = [ancilla_qubits[i] if y[i]=='1' for i in range(len(y))]
                
                R = _tools.z_pauli_on_qubits(subset_with_z, qubit_labels) #Z pauli to be measured
            
                assert(_tools.is_stabilizer(R, inverse_tableau)), "Z should be a stabilizer of the ancilla qubits!"
                if not P.commutes(R):
                        if not Q.commutes(R):
                            contribution = -2 * _tools.is_stabilizer_or_antistabilizer(Q * P, inverse_tableau) 
                            #if contribution != 0:
                            deltas[(P, Q ,y)] = contribution

    return deltas

def pauli_support(p):
    #computes combinatorial factors
    return [i for i in range(len(p)) if p[i] != 0]

def pauli_xy_support(p):
    #computes combinatorial factors
    return [i for i in range(len(p)) if p[i] != 0 and p[i] != 3]

def compute_contribution(eeg, rate, det_pauli, tableau):
    contribution = rate*alpha_pauli(eeg, tableau, det_pauli)/2 
    return contribution
    
def generate_dem(dets_as_pauli_strings, eoc_eeg, sim, n_rounds=None, n_logical=None, negate_det = []):
    #note: we're checking which detectors are affected, then combining the results to get which DEM event is affected. 
    #let's try to do this without the sensitivity vectors to make it easier to generalize

    #one way to think about this is that we group the errors by DEM event, then estimate the rate to first order. 
    #I don't think that we can guarantee that going to higher guarantees the same effect. 
    
    #simply go through the list of Pauli strings and compute which ones are affected. 
    #I'm guessing multiple detectors can be affected in this case.
    def to_detectors(x_eff, n_rnd, n_log):
        x_logical = x_eff[-1*n_log:]
        x_eff = x_eff[:-1*n_log]
        n_aux = len(x_eff)//n_rnd
        x = x_eff[:n_aux]
        for i in range(1,n_rnd):
            x += xor_binary_strings(x_eff[(i-1)*n_aux:i*n_aux], x_eff[i*n_aux:(i+1)*n_aux])
        x = x+x_logical
        return x 

    dem = defaultdict(float)
    eoc_eeg_terms = eoc_eeg.keys()
    sca_eegs = [t for t in eoc_eeg_terms if t.errorgen_type != 'H']
    dem_string = '0'*len(dets_as_pauli_strings) #initialize event bitstring
    for eeg in sca_eegs:
        weight = eoc_eeg[eeg]
        dets_fired = []
        test_contributions = []
        for i, det_pauli in enumerate(dets_as_pauli_strings):
            contribution = compute_contribution(eeg, weight, det_pauli, sim)
            if contribution != 0 and i not in negate_det: #note: I hard-coded in i != 20 for msc. 
                dets_fired.append(i)
                #print(eeg, i, contribution)
                test_contributions.append(contribution)
            if i in negate_det and contribution==0:
                #print(eeg)
                dets_fired.append(i)
            else: 
                test_contributions.append(contribution)
        det_string = ''.join('1' if i in dets_fired else '0' for i in range(len(dets_as_pauli_strings))) 
        if len(dets_fired) > 0: #hack
            contribution = compute_contribution(eeg, weight, dets_as_pauli_strings[dets_fired[0]], sim)
            #print(dets_fired, test_contributions)
        dem[det_string] += -1*contribution #clearly false
    #convert this to a detector string
    h_terms = [t for t in eoc_eeg_terms if t.errorgen_type == 'H']
    terms_by_effect = terms_by_effect = sort_terms_by_effect(h_terms, dets_as_pauli_strings, sim)   #make this faster by looking at XY support on ancilla Qs
    for k, eeglist in terms_by_effect.items():   
        for (eeg1, eeg2) in itertools.product(eeglist, repeat=2):
            weight = eoc_eeg[eeg1]*eoc_eeg[eeg2]/2 
            #TODO gives you an S or C term. Figure out which one
            dets_fired = []
            for i, det_pauli in enumerate(dets_as_pauli_strings):
                #print(i, det_pauli)
                new_eeg = LocalStimErrorgenLabel('C',(eeg1.basis_element_labels[0], eeg2.basis_element_labels[0]))
                #print(new_eeg)
                contribution = compute_contribution(new_eeg, weight, det_pauli, sim)
                if contribution != 0: 
                    dets_fired.append(i)
            det_string = ''.join('1' if i in dets_fired else '0' for i in range(len(dets_as_pauli_strings))) 
            if len(dets_fired) > 0: 
                contribution = compute_contribution(new_eeg, weight, dets_as_pauli_strings[dets_fired[0]], sim)
                
                dem[det_string] += -1*contribution
    return dem

def to_full_pauli(partial_pauli, qs, n_qubits):
    full_p = stim.PauliString('I'*n_qubits)
    for i,q in enumerate(qs):
        full_p[q]=partial_pauli[i]
    return full_p
    

def get_detector_as_parity(detector_indices, measurements, n_qubits):
    #for each measurement in the detector, 
    p_net = stim.PauliString('I'*n_qubits)
    for idx in detector_indices:
        #turn each Pauli into an n-qubit Pauli
        partial_pauli, qs = measurements[idx]
        p = to_full_pauli(partial_pauli, qs, n_qubits)
        p_net *= p
    #take the product of these Paulis
    return p_net

def sort_terms_by_effect(terms, detectors, sim, show_progress=False):
    sorted_terms = defaultdict(list)
    for j,eeg in enumerate(terms):
        dets_fired = []
        #print(eeg[0])
        if show_progress:
            if j%100==0:
                print(j)
        for i, det_pauli in enumerate(detectors):
            #print(i, det_pauli)
            #print(eeg[0].basis_element_labels)
            #contribution = dems.compute_contribution(LocalStimErrorgenLabel('S',eeg[0].basis_element_labels), 1, det_pauli, tableau)
            P = eeg.basis_element_labels[0]
            if not P.commutes(det_pauli): 
                dets_fired.append('1')
            else: dets_fired.append('0')
        det_string = ''.join(dets_fired) 
        #print(eeg[0], det_string)
        sorted_terms[det_string].append(eeg)

    return dict(sorted_terms)

def streamlined_create_eoc_h_dem(h_egp, circuit, detectors, sim, theta, parameter_indexing, include_spam=False, show_progress=False):
    #propagate to EOC
    #sort terms by their detector effects
    #compute effects for pairs with the same effect
    dem = Counter({})
    
    mapping_to_eoc_eeg = h_egp.errorgen_transform_map(circuit, include_spam=include_spam)
    eeg_terms = list(set(mapping_to_eoc_eeg.values()))
    
    terms_by_effect = sort_terms_by_effect(eeg_terms, detectors, sim) 
    if show_progress:
        print('sorted terms by effect')
    svs = _svs.compute_sensitivity_vectors(mapping_to_eoc_eeg, h_egp, circuit, parameter_indexing, return_s_equivalent=False)
    for effect, eeglist in terms_by_effect.items():   
        effect_detectors = [(i,detectors[i]) for i in range(len(effect)) if effect[i]=='1']
        #print(effect_detectors)
        if show_progress:
            print(f'iterating though list of size {len(eeglist)} ({len(eeglist)**2}) pairs with {len(effect_detectors)} detectors')
        if len(effect_detectors)>0:
            for j, (eeg1, eeg2) in enumerate(itertools.product(eeglist, repeat=2)):
                #if j%1000==0: print(j)
                sv1 = svs[eeg1[0]]
                sv2 = svs[eeg2[0]]
                n_params = len(theta)
                M = _svs.create_sensitivity_matrix(sv1, sv2, n_params)
                weight = np.dot(theta, np.dot(M, theta))/2 #eeg1[1]*eeg2[1]*
                #TODO gives you an S or C term. Figure out which one
                dets_fired = []
                
                ####TODO: effect tells us precisely what detectors need to be looked at. We know that other detectors won't be affected
                ####So actually all we need to do here is check the impact of these C terms on those detectors
                ####it should actually be that we can check the effect on just one of the detectors. 
                for i, det_pauli in effect_detectors:
                    #print(i, det_pauli)
                    new_eeg = LocalStimErrorgenLabel('C',(eeg1[0].basis_element_labels[0], eeg2[0].basis_element_labels[0]))
                    #print(new_eeg)
                    contribution = compute_contribution(new_eeg, weight, det_pauli, sim)
                    if contribution != 0: 
                        dets_fired.append(i)
                det_string = ''.join('1' if i in dets_fired else '0' for i in range(len(detectors))) 
                if len(dets_fired) > 0: #hack
                    contribution = compute_contribution(new_eeg, weight, detectors[dets_fired[0]], sim)
                    #print(dets_fired, contribution)
                    dem[det_string] += -1*contribution

    return dem

def split_generator(term_sorting, eoc_eeg):
    generator_split = []
    base_events = []
    for k, v in term_sorting.items():
        term_dict = {}
        base_events.append(k)
        for egen in v:
            r = eoc_eeg[egen]
            term_dict[egen] = r
        generator_split.append(term_dict)
    return generator_split, base_events

def add_to_dem(dem, base_events, leading_order_channels, eoc_eeg, dets_as_pauli_strings, sim):
    for event, egens in zip(base_events, leading_order_channels):
        if event != '0'*len(event):
            if all(k.errorgen_type=='S' for k in egens.keys()) or (len(egens.keys())==1 and all(k.errorgen_type=='H' for k in egens.keys())):
                first_flipped_detector = event.find('1')
                det_pauli = dets_as_pauli_strings[first_flipped_detector]
                contribution = estimate_error_rate_taylor(egens, sim, det_pauli) #maker sure constant factors are correct)
                dem[event] += contribution
            else:
                #properly deal with H terms by separating SCA terms from H terms
                h_terms = [t for t in egens if t.errorgen_type == 'H']
                sca_terms = [t for t in egens if t.errorgen_type != 'H']

                for (eeg1, eeg2) in itertools.product(h_terms, repeat=2):
                    weight = eoc_eeg[eeg1]*eoc_eeg[eeg2]/2  
                    #pick out one of the detectors flipped
                    first_flipped_detector = event.find('1')
                    det_pauli = dets_as_pauli_strings[first_flipped_detector]

                    new_eeg = LocalStimErrorgenLabel('C',(eeg1.basis_element_labels[0], eeg2.basis_element_labels[0]))
                    contribution = compute_contribution(new_eeg, weight, det_pauli, sim)
                    dem[event] += -1*contribution.real

                #TODO handle SCA terms
                for eeg in sca_terms:
                    weight = eoc_eeg[eeg]
                    first_flipped_detector = event.find('1')
                    det_pauli = dets_as_pauli_strings[first_flipped_detector]
                    contribution = compute_contribution(eeg, weight, det_pauli, sim)
                    dem[event] += -1*contribution.real

    return dem

#TODO: test
def compose_dems(dem1, dem2):
    #composes two dems
    new_dem = defaultdict(float)
    events = list(set(dem1.keys()).union(set(dem2.keys())))
    for k in events:
        if k in dem1:
            if k in dem2:
                new_dem[k] = dem1[k]*(1-dem2[k])+(1-dem1[k])*dem2[k]
            else: new_dem[k] = dem1[k]
        else: new_dem[k] = dem2[k]
    return new_dem

def generate_dem_higher_order(dets_as_pauli_strings, eoc_eeg, sim,  zassenhaus_order=1, add_type='add'):
    #note: we're checking which detectors are affected, then combining the results to get which DEM event is affected. 
    #let's try to do this without the sensitivity vectors to make it easier to generalize

    #one way to think about this is that we group the errors by DEM event, then estimate the rate to first order. 
    #I don't think that we can guarantee that going to higher guarantees the same effect. 
    
    #simply go through the list of Pauli strings and compute which ones are affected. 
    #I'm guessing multiple detectors can be affected in this case.

    dem = defaultdict(float)
    eoc_eeg_terms = eoc_eeg.keys()
    term_sorting = sort_terms_by_effect(eoc_eeg, dets_as_pauli_strings, sim)
    generator_split, base_events = split_generator(term_sorting, eoc_eeg)

    terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=zassenhaus_order) #TODO: edit this function to return a list by zassenhaus order
    leading_order_channels = terms_zassenhaus[:(-1)*(zassenhaus_order-1)]
    higher_order_channels = terms_zassenhaus[(-1)*(zassenhaus_order-1):]

    dem = add_to_dem(dem, base_events, leading_order_channels, eoc_eeg, dets_as_pauli_strings, sim)
    for channel in higher_order_channels:
        #sort by dem event
        term_sorting = sort_terms_by_effect(channel, dets_as_pauli_strings, sim)
        generator_split, new_events = split_generator(term_sorting, channel)
        terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=1) #TODO generalize. 
        if add_type=='add':
            dem = add_to_dem(dem, new_events, terms_zassenhaus, channel, dets_as_pauli_strings, sim) #TODO think about if adding is correct here. 
        elif add_type=='compose':
            additional_dem = defaultdict(float)
            additional_dem = add_to_dem(additional_dem, new_events, terms_zassenhaus, channel, dets_as_pauli_strings, sim)
            #print(dem, additional_dem)
            dem = compose_dems(dem, additional_dem)
        else:
            raise('add_type not recognized')
    
    #TODO figure out a way to do iterative adjustment of the DEM based upon non-S errors existing in different DEM event channels
    return dem

def bulk_approximate_pauli_expectation(eoc_err, pyg_c, tableau, paulis, order=1, expanded_error=None, truncation_threshold=1e-10):
    #compute expanded error
    if expanded_error is None:
        expanded_error = eprop.error_generator_taylor_expansion(eoc_err, order=order)
    #compute expectation value for each pauli in paulis
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    pauli_corrections = []
    for pauli in paulis:
        print(pauli)
        #this is the code from the pygsti implementation of the correction.     
        correction = 0
        
        #labels_by_order = [list(errorgen_dict.keys()) for errorgen_dict in expanded_error]
        # Get a similar structure for the corresponding rates
        #rates_by_order = [list(errorgen_dict.values()) for errorgen_dict in expanded_error]
            
        for egen_dict in expanded_error:   
            alpha_errgen_prods = np.zeros(len(egen_dict))
            for i, (lbl, rate) in enumerate(egen_dict.items()):
                if abs(rate) > truncation_threshold:
                    sensitivity = dems.alpha_pauli(lbl, sim, pauli)
                    alpha_errgen_prods[i] = np.real_if_close(sensitivity*rate)
            correction += np.sum(alpha_errgen_prods)
        pauli_corrections.append(correction)

    return pauli_corrections
    
#TODO test below
def estimate_error_rate_taylor(edict, sim, det_pauli, order=1, truncation_threshold=1e-9):
    '''
    function to estimate DEM event rate for a single-DEM-event channel 
    specified as a dictionary of elementary error generators and their rates.
    Computes exact rate for special cases, otherwise Taylor expands the error to a specified order (default 1)'''
    #if only S errors: compute exact
    if all(k.errorgen_type=='S' for k in edict.keys()):
        contribution = (1-np.exp(-2*sum(edict.values())))/2 #maker sure constant factors are correct)
        #print(sum(edict.values()))
    #if only one H error: compute exact
    elif len(edict.keys())==1 and all(k.errorgen_type=='H' for k in edict.keys()):
        #TODO verify
        contribution = np.sin(sum(edict.values()))**2 #only one value
    #else: compute with a small-order taylor expansion
    else:
        contribution = 0
        expanded_error = eprop.error_generator_taylor_expansion(edict, order=order)
        for egen_dict in expanded_error:   
            alpha_errgen_prods = np.zeros(len(egen_dict))
            for i, (lbl, rate) in enumerate(egen_dict.items()):
                if abs(rate) > truncation_threshold:
                    sensitivity = alpha_pauli(lbl, sim, det_pauli)
                    alpha_errgen_prods[i] = np.real_if_close(sensitivity*rate)
            contribution += np.abs(np.sum(alpha_errgen_prods))
        #print(alpha_errgen_prods)
        contribution /= 2 #convert from change in expextation value to probability
    return contribution

#how do you determine which products of single-event channels need to be dealt with?
#you xor the events together to see if you get 0. 
def find_emergent_events(events, order_upto):
    #take combinations of up to order DEM events
    #check if they sum to 0
    emergent_event_combos = []
    return emergent_event_combos

def generate_higher_order_correction(event, dem_event_channels, sim, init_dem, dets, order=2, truncation_threshold=1e-9):
    '''
    generates corrections to a DEM based on higher-order emergent terms 
    in the composition of single-dem-event channels

    returns: correction_dem (dict) A dictionary representing a correction to the rates of DEM events
    '''
    #correction_dem = Counter({})
    #TODO taylor expand product of dem_event channels to order
    expanded_error = taylor_expand_error(dem_event_channels) #TODO figure out how to actually do this
    #TODO compute difference between DEM-predicted polarizations and those predicted by Taylor expansion
    pauli = event_to_pauli(event, dets)
    val_higher_order = eprop.approximate_pauli_expecation_value(pauli, expanded_error, sim)
    val_dem = compute_dem_event_probability(event, init_dem) #TODO

    #TODO take the difference between expected and observed and W-H to get corrections
    dem_err = higher_order_predictions - dem_predictions

    #TODO should probably also correct the constituent events that built the emergent event

    return dem_err

# def generate_fourth_order_h_corrections(events)
#     '''
#     specialized function to generate fourth order corrections from h-only models
#     '''
#     correction_dem = Counter()
#     candidate_events = find_emergent_events(events, order_upto=4) #find all possible emergent events
#     for emergent_event in candidate_events:
#         event_rate = generate_higher_order_correction(dem_event_channels, sim, dem_events, order=2, truncation_threshold=1e-9)
#         correction_dem[event_rate] += event_rate
#     return correction_dem