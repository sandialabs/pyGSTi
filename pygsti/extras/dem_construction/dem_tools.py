import numpy as np
import itertools
import stim

from collections import defaultdict
import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel

#Note on code integration process: This was written to pass around the TableauSimulator to avoid computing it multiple times. 
# But pyGSTi's bulk_alpha_pauli operates on a Tableau. 
# So this should all be rewritten to take in the stim Tableau and use bulk_alpha_pauli as efficiently as possible

def pauli_product(P1, P2):
    P3 = P1*P2
    return (P3.sign, P3 / P3.sign)

def format_dem_stim(dem_dict, n_logical):
    '''
    formats a dictionary of event probabilities into a stim dem
    '''
    n_detectors = len(list(dem_dict.keys())[0])
    #detectors called D0,D1,...D(n_detectors-1)
    #need to add in logical detectors as a separate thing
    dem_string = ''
    for dbs, prob in dem_dict.items():
        if prob > 0:
            line = f'error({prob}) '+''.join([f'D{i} ' if dbs[i]=='1' else '' for i in range(n_detectors-n_logical)]+[f'L{i} ' if dbs[n_detectors-n_logical+i]=='1' else '' for i in range(n_logical)])+'\n'
            dem_string+=line
    return dem_string

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

def compute_contribution(eeg, rate, det_pauli, tableau):
    contribution = rate*eprop.bulk_alpha_pauli([eeg], tableau, [det_pauli])/2 #TODO improve since this'll be slow
    return contribution

def compute_contribution_bulk(eegs, rates, det_pauli, tableau):
    alpha_paulis = eprop.bulk_alpha_pauli(eegs, tableau, [det_pauli])/2 #TODO improve since this'll be slow
    contributions = [r*ap for r, ap in zip(rates, alpha_paulis)]  
    return contributions

def sort_terms_by_effect(terms, detectors, show_progress=False):
    sorted_terms = defaultdict(list)
    for j,eeg in enumerate(terms):
        dets_fired = []
        #print(eeg[0])
        if show_progress:
            if j%100==0:
                print(j)
        for i, det_pauli in enumerate(detectors):
            P = eeg.basis_element_labels[0]
            if not P.commutes(det_pauli): 
                dets_fired.append('1')
            else: dets_fired.append('0')
        det_string = ''.join(dets_fired) 
        sorted_terms[det_string].append(eeg)

    return dict(sorted_terms)

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

def add_to_dem(dem, base_events, leading_order_channels, eoc_eeg, dets_as_pauli_strings, tableau): 
    for event, egens in zip(base_events, leading_order_channels):
        if event != '0'*len(event):
            first_flipped_detector = event.find('1')
            det_pauli = dets_as_pauli_strings[first_flipped_detector]

            if all(k.errorgen_type=='S' for k in egens.keys()) or (len(egens.keys())==1 and all(k.errorgen_type=='H' for k in egens.keys())):
                contribution = estimate_error_rate_taylor(egens, tableau, det_pauli) #maker sure constant factors are correct)
                dem[event] += contribution
            else:
                #properly deal with H terms by separating SCA terms from H terms
                h_terms = [t for t in egens if t.errorgen_type == 'H']
                sca_terms = [t for t in egens if t.errorgen_type != 'H']

                h_weights = []
                h_product_eegs = []

                for (eeg1, eeg2) in itertools.product(h_terms, repeat=2):
                    weight = eoc_eeg[eeg1]*eoc_eeg[eeg2]/2  
                    h_weights.append(weight)
                    
                    #pick out one of the detectors flipped
                    new_eeg = LocalStimErrorgenLabel('C',(eeg1.basis_element_labels[0], eeg2.basis_element_labels[0]))
                    h_product_eegs.append(new_eeg)

                weights = h_weights+[eoc_eeg[eeg] for eeg in sca_terms]
                all_terms = h_product_eegs + sca_terms
                contributions = compute_contributions_bulk(all_terms, weights, det_pauli, tableau)
                dem[event] += -1*np.sum(np.real(np.array(contributions)))

    return dem

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

def generate_dem_higher_order(dets_as_pauli_strings, eoc_eeg, sim,  zassenhaus_order=1):
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

    terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=zassenhaus_order)
    leading_order_channels = terms_zassenhaus[:(-1)*(zassenhaus_order-1)]
    higher_order_channels = terms_zassenhaus[(-1)*(zassenhaus_order-1):]

    dem = add_to_dem(dem, base_events, leading_order_channels, eoc_eeg, dets_as_pauli_strings, sim)
    for channel in higher_order_channels:
        #sort by dem event
        term_sorting = sort_terms_by_effect(channel, dets_as_pauli_strings, sim)
        generator_split, new_events = split_generator(term_sorting, channel)
        terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=1) 

        additional_dem = defaultdict(float)
        additional_dem = add_to_dem(additional_dem, new_events, terms_zassenhaus, channel, dets_as_pauli_strings, sim)
        dem = compose_dems(dem, additional_dem)
    
    return dem
    
def estimate_error_rate_taylor(edict, tableau, det_pauli, order=1, truncation_threshold=1e-9):
    '''
    function to estimate DEM event rate for a single-DEM-event channel 
    specified as a dictionary of elementary error generators and their rates.
    Computes exact rate for special cases, otherwise Taylor expands the error to a specified order (default 1)'''
    #if only S errors: compute exact
    if all(k.errorgen_type=='S' for k in edict.keys()):
        contribution = (1-np.exp(-2*sum(edict.values())))/2 
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
            egens = np.array(egens.keys())
            rates = np.array(egens.values())
            sensitivities = eprop.bulk_alpha_pauli(egens, tableau, det_pauli)
            alpha_errgen_prods = np.real_if_close(sensitivities) @ rates
            contribution += np.abs(np.sum(alpha_errgen_prods))
        contribution /= 2 
    return contribution