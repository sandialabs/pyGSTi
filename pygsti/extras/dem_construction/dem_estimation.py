#functions for estimation a DEM from data.
#I think I have some of this in old notebooks...

import numpy as np

def compute_dem_pols_from_z_exps(z_exps, k_max, n_aux, n_rounds):
    '''
    exhaustively compute z polarizations of weight < k_max and put into a dictionary
    n_aux is the number of aux qubits (in one round of syndrome extraction
    '''
    dem_pols = {}
    event_bitstrings = up_to_kbits((n_rounds-1)*n_aux, k_max) #note: these are currently strings
    for bs in event_bitstrings: 
        #bs is a DEM string. We need to translate it into an effective syndrome bit string
        #TODO
        bs_eff = bs[:n_aux]+ ''.join(xor_binary_strings(bs[n_aux*i: n_aux*(i+1)], bs[n_aux*(i+1): n_aux*(i+2)]) for i in range(0,n_rounds-2))+bs[-n_aux:]
        #print(bs_eff)
        qubit_subset = [q for i, q in enumerate(ancilla_qubits) if bs_eff[i]=='1']
        pauli_eff_ancilla = ''.join(['Z' if bs_eff[i]=='1' else 'I' for i in range(len(bs_eff))])
        pauli_eff = "".join(i + j for i, j in itertools.zip_longest('I'*(n_aux+1), pauli_eff_ancilla, fillvalue=''))

        #print(bs, bs_eff, qubit_subset)

        dem_pols[bs] = z_exps[pauli_eff]

    return dem_pols

def compute_complete_dem(rounds, dem_pols, ancilla_qubits):
    #iterate through events and compute probs
    n_aux = len(ancilla_qubits)//rounds
    n_detectors = (rounds-1)*n_aux

    normalization = 2**(1-n_detectors)
    dem_strings = up_to_kbits((rounds)*n_aux, (rounds)*n_aux)[1:]

    #print(dem_strings)

    dem = {event: compute_dem_prob(rounds, event, dem_pols, ancilla_qubits, k_max=None) for event in dem_strings}
    
    return dem