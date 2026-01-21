import numpy as np
import stim

from . import dem_tools as _dem
#from helper_functions import *

def postselect_error_channel(eoc_eeg_terms, qubit_labels, inverse_tableau, ancilla_qubits, n_rounds, n_aux, svs, theta):
    #TODO find all leading-order terms with no contribution to DEM events
    postselected_error = []
    for eeg in eoc_eeg_terms:
        sv = svs[eeg[0]]
        sv = svs[eeg[0]]
        if eeg[0].errorgen_type=='H':
            P = eeg[0].basis_element_labels[0]
            contribution = {} #auto-add 1st order H terms
        if eeg[0].errorgen_type=='S':
            P = eeg[0].basis_element_labels[0]
            contribution = _dem.compute_s_term_dem(P, qubit_labels, inverse_tableau, ancilla_qubits, n_rounds, n_aux, sv, theta)
        if eeg[0].errorgen_type=='C':
            P = eeg[0].basis_element_labels[0]
            Q = eeg[0].basis_element_labels[1]
            contribution = _dem.compute_c_term_dem(P, Q, qubit_labels, inverse_tableau, ancilla_qubits, n_rounds, n_aux, sv, theta)
        if eeg[0].errorgen_type=='A':
            P = eeg[0].basis_element_labels[0]
            Q = eeg[0].basis_element_labels[1]
            contribution = _dem.compute_a_term_dem(P, Q, qubit_labels, inverse_tableau, ancilla_qubits, n_rounds, n_aux, sv, theta)
        if contribution == {} or list(contribution.keys())==['0'*n_rounds*n_aux]: ###TEMPORARY HACK TO FIND CASES WITH EFFECTIVELY NO CONTRIBUTION
            postselected_error.append(eeg)
            #keep. check that there aren't any cases where the all 0s string is in the dictionary
    #convert this to a detector string

    #it'd be nice to include the S terms here in the logical error channel
    # h_terms = [t for t in eoc_eeg_terms if t[0].errorgen_type == 'H']
    # for (eeg1, eeg2) in itertools.product(eeglist, repeat=2):
    #     for i, det_pauli in enumerate(dets_as_pauli_strings):
    #         contribution = compute_contribution(eeg, det_pauli)
    #         if contribution == {}: 
                #keep the corresponding S/C errors in the model
    return postselected_error

def check_logical_effect_hs(full_pauli, data_qs, logicals, stabilizers):
    #check Pauli indices of the error channels to see if they correspond to logical Paulis (up to multiplication by stabilizers)
    #Check the Paulis against X_L, Y_L, Z_L and those times stabilizers
    pauli_ordering = ['I', 'X', 'Y', 'Z']
    P = ''.join(pauli_ordering[full_pauli[d]] for d in data_qs) #Pauli, restricted to data qubits
    if P in logicals:
        #print(P, logicals[P])
        return logicals[P]
    elif P=='I'*len(data_qs):
        return None
    else:
        #print('here',P)
        inv_tableau = stabilizers.inverse()
        p2 = inv_tableau(stim.PauliString(P))
        if p2.sign == +1 and "X" not in str(p2) and "Y" not in str(p2):
            return None
        #check for equivalence up to stabilizers
        #this is clearly broken
        else:
            for phys_P, log_P in logicals.items(): 
                P = stim.PauliString(P)
                #print(P*stim.PauliString(phys_P))
                p2 = inv_tableau(P*stim.PauliString(phys_P))
                if p2.sign == +1 and "X" not in str(p2) and "Y" not in str(p2):
                    #print('logical up to stabilizer', P, P*stim.PauliString(phys_P)) #print the Pauli on the data qubits and the stabilizer/product of stabilizers that need to be multiplied by to get the logical operator as-stated.
                    return log_P
        return None


def estimate_logical_fidelity(list_of_eegs, sensitivity_vectors, data_qs, logicals, stabilizers_tableau):
    pauli_ordering = ['I', 'X', 'Y', 'Z']
    inf = 0
    logical_error_terms = []
    for eeg in list_of_eegs:
        if eeg[0].errorgen_type in ["H", "S"]:
            P = eeg[0].basis_element_labels[0]
            #We should be looking at exclusively errors on the logical subsystem. 
            logical_effect = check_logical_effect_hs(P, data_qs, logicals, stabilizers_tableau) 
            if logical_effect is not None:
                logical_error_terms.append((eeg, logical_effect))
                #TODO maybe here we want to look at a higher-order expansion of the remaining generators
                sv1 = sensitivity_vectors[eeg[0]]
                if eeg[0].errorgen_type=='S':
                    rate = np.dot(sv1, theta)
                    inf += rate
                    #print('S', ''.join(pauli_ordering[P[d]] for d in data_qs), rate, sv1, theta)
                elif eeg[0].errorgen_type=='H':
                    sign = eeg[1]
                    M = sign*create_sensitivity_matrix(sv1, sv1, len(theta))
                    rate = np.dot(theta, np.dot(M, theta))
                    #print('H', ''.join(pauli_ordering[P[d]] for d in data_qs), rate)
                    inf += rate
    return 1-inf, logical_error_terms

def compute_logical_error_channel(err, data_qs, logicals, stabilizers):
    #iterate through each term
    #determine its logical effect
    log_channel = {}
    for eeg, rate in err.items():
        log_P = check_logical_effect_hs(eeg.basis_element_labels[0], data_qs, logicals, stabilizers)
        if log_P is not None:
            #add to that part of the error channel
            if (eeg.errorgen_type, str(log_P)) in log_channel:
                log_channel[(eeg.errorgen_type, str(log_P))].append((eeg, rate))
            else:
                log_channel[(eeg.errorgen_type, str(log_P))] = [(eeg, rate)]
    return log_channel
    
def build_dem_logical_part(dem_contributing, sensitivity_vectors, theta, data_qubits, stabilizers_tableau, logicals, order=3):
    dem_logical = {}
    for k, termlist in dem_contributing.items():
        #turn this termlist into an EOC EEG
        eoc_err = {}
        print(f'creating new channel with {len(termlist)} terms')
        for term in termlist:
            sv = sensitivity_vectors[term[0]]
            rate = np.dot(sv,theta)
            eoc_err[term[0]] = rate*term[1] #rate*sign
        #expand to sufficiently high order --> maybe not needed?
        #returns a list of dictionaries, presumably for each order. 
        taylor_expanded_error = eprop.error_generator_taylor_expansion(eoc_err, order=order) #slow
        err = Counter({})
        for d_order in taylor_expanded_error:
            err.update(d_order)
        #compute logical effects
        logical_error = compute_logical_error_channel(err, data_qubits, logicals, stabilizers_tableau)
        dem_logical[k] = logical_error
    return dem_logical