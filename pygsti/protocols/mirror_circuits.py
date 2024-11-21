import numpy as _np
import tqdm as _tqdm

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.protocols.protocol import FreeformDesign as _FreeformDesign
from pygsti.protocols.protocol import CombinedExperimentDesign as _CombinedExperimentDesign

from pygsti.processors import central_pauli as _cp
from pygsti.processors import random_compilation as _rc

import utils #TODO: integrate all the needed functionality from utils into pygsti

#TODO: OOP-ify this code?

## FUNCTIONS TAKEN AND MODIFIED FROM utils.py in circuit-verification-using-mirroring
# Modifications are mostly:
# 1. Don't str() args
# 2. Use a rand_state for deterministic edesign
def bare_mirror_circuit(circ, randomized_state_preparation=True, rand_state=None, pad_layers=True):
    qubits = circ.line_labels
    n = circ.width
    d = circ.depth
    
    layers_to_invert = [utils.haar_random_u3_layer(qubits, rand_state)] if randomized_state_preparation else []
    for j in range(d):
        layer = circ.layer(j)
        if pad_layers and layer[0].name == 'Gu3':
            # Pad single-qubit layers to be dense
            layer = utils.pad_layer(circ.layer(j), qubits)
        layers_to_invert.append(layer)
    
    mirrored_layers = []
    for layer in layers_to_invert:
        inverse_layer = [utils.gate_inverse(gate_label) for gate_label in layer]
        mirrored_layers.insert(0, inverse_layer)

    # Avoid checks for speed
    mc = _Circuit(layers_to_invert + mirrored_layers, line_labels=qubits, check=False, expand_subcircuits=False)

    return mc, '0'*n


## For legacy reasons
def sample_central_pauli_mirror_circuits(edesign, num_ref_per_width=100, num_mcs_per_circ=100,
    randomized_state_preparation=True, rand_state=None, verbose=True):
    return sample_mirror_circuits(edesign, central_pauli=True, random_compiling=False,
                                  num_ref_per_width=num_ref_per_width, num_mcs_per_circ=num_mcs_per_circ,
                                  randomized_state_preparation=randomized_state_preparation,
                                  rand_state=rand_state,
                                  #verbose=verbose
                                  )

def sample_mirror_circuits(edesign, central_pauli=False, random_compiling=True,
                           num_ref_per_width=100, num_mcs_per_circ=100,
                           randomized_state_preparation=True, rand_state=None):
    if rand_state is None:
        rand_state = _np.random.RandomState()

    rc_rc_mcs = {}
    bare_rc_mcs = {}
    #bare_bare_mcs = {}
    central_pauli_mcs = {}
    qubit_subsets = {}

    for c, auxlist in _tqdm.tqdm(edesign.aux_info.items(), ascii=True, desc='Sampling mirror circuits'):
        if isinstance(auxlist, dict):
            auxlist = [auxlist]
        
        aux = auxlist[0]
        if aux['width'] not in qubit_subsets:
            qubit_subsets[aux['width']] = []
        qubit_subsets[aux['width']].append(c.line_labels)

        for j in range(num_mcs_per_circ):
            if central_pauli:
                cp_mc, cp_bs = _cp.central_pauli_mirror_circuit(c, randomized_state_preparation=randomized_state_preparation,
                                                            rand_state=rand_state)
                cp_mc_aux = [{'base_aux': a, 'idealout': cp_bs, 'id': j} for a in auxlist]
                central_pauli_mcs[cp_mc] = cp_mc_aux
            if random_compiling:
                mc, bs = bare_mirror_circuit(c, randomized_state_preparation, rand_state=rand_state, pad_layers=True)
                #mc_aux = [{'base_id': a['id'], 'idealout': bs, 'id': j} for a in auxlist]
                #bare_bare_mcs[mc] = mc_aux 

                rc_rc_mc, rc_rc_bs = _rc.pauli_randomize_circuit(mc, rand_state=rand_state)
                half_rc_mc, bare_rc_bs = _rc.pauli_randomize_circuit(mc[mc.depth // 2:], rand_state=rand_state)
                bare_rc_mc = mc[:mc.depth // 2] + half_rc_mc
            
                bare_rc_mc_aux = [{'base_aux': a, 'idealout': bare_rc_bs, 'id': j} for a in auxlist]
                bare_rc_mcs[bare_rc_mc] = bare_rc_mc_aux 

                rc_rc_mc_aux = [{'base_aux': a, 'idealout': rc_rc_bs, 'id': j} for a in auxlist]
                rc_rc_mcs[rc_rc_mc] = rc_rc_mc_aux

    ref_cs = {}
    cp_ref_cs = {}
    for w, width_subsets in qubit_subsets.items():
        subset_indices = rand_state.choice(list(range(len(width_subsets))), num_ref_per_width)
        for j in _tqdm.tqdm(range(len(subset_indices)), ascii=True, desc=f'Sampling width {w} reference circuits'):
            subset_idx = subset_indices[j]
            if central_pauli:
                cp_ref_circ, cp_ref_bs = _cp.central_pauli_mirror_circuit(_Circuit('',
                                                                    line_labels=width_subsets[subset_idx]), 
                                                                    randomized_state_preparation=randomized_state_preparation,
                                                                    rand_state=rand_state)
                if cp_ref_circ in cp_ref_cs:
                    cp_ref_cs[cp_ref_circ].append({'idealout': cp_ref_bs, 'id': j, 'width': w})
                else:
                    cp_ref_cs[cp_ref_circ] = [{'idealout': cp_ref_bs, 'id': j, 'width': w}]
            if random_compiling:
                prep_layer = utils.haar_random_u3_layer(width_subsets[subset_idx])
                ref_circ = _Circuit([prep_layer], width_subsets[subset_idx])
                inverse_layer = [utils.gate_inverse(label) for label in prep_layer]
                ref_circ = ref_circ + _Circuit([inverse_layer], width_subsets[subset_idx])
                ref_circ, bs = _rc.pauli_randomize_circuit(ref_circ)

                if ref_circ in ref_cs:
                    ref_cs[ref_circ].append({'idealout': bs, 'id': j, 'width': w})
                else:    
                    ref_cs[ref_circ] = {'idealout': bs, 'id': j, 'width': w}

    edesigns = {}
    if random_compiling:
        edesigns['ref'] = _FreeformDesign(ref_cs)
        #edesigns['bb'] = _FreeformDesign(bare_bare_mcs) # Not needed for RC fid estimates
        edesigns['br'] = _FreeformDesign(bare_rc_mcs)
        edesigns['rr'] = _FreeformDesign(rc_rc_mcs)
    if central_pauli:
        edesigns['cp'] = _FreeformDesign(central_pauli_mcs)
        edesigns['cpref'] = _FreeformDesign(cp_ref_cs)

    return _CombinedExperimentDesign(edesigns, skip_writing_all_circuits=True)