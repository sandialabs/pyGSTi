import numpy as np
import itertools

from collections import defaultdict
import pygsti.tools.errgenproptools as eprop
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel

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
        #print(len(dbs))
        if prob > 0:
            line = f'error({prob}) '+''.join([f'D{i} ' if dbs[i]=='1' else '' for i in range(n_detectors-n_logical)]+[f'L{i} ' if dbs[n_detectors-n_logical+i]=='1' else '' for i in range(n_logical)])+'\n'
            dem_string+=line
    return dem_string


def compute_contribution(eeg, rate, det_pauli, tableau):
    contribution = rate*alpha_pauli(eeg, tableau, det_pauli)/2 
    return contribution

def sort_terms_by_effect(terms, detectors, sim=None, show_progress=False):
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