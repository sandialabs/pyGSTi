from __future__ import annotations
from typing import Callable, Union, Literal, Optional, Tuple

import numpy as _np
import tqdm as _tqdm

from collections import defaultdict

from pygsti.circuits.circuit import Circuit as _Circuit

from pygsti.protocols.protocol import FreeformDesign as _FreeformDesign
from pygsti.protocols.protocol import CombinedExperimentDesign as _CombinedExperimentDesign

from pygsti.processors import random_compilation as _rc

#TODO: OOP-ify this code?

def make_mirror_edesign(test_edesign: _FreeformDesign,
                        ref_edesign: Optional[_FreeformDesign] = None,
                        id_exact_circ_dict: Optional[dict] = None,
                        num_mcs_per_circ: int = 100,
                        num_ref_per_qubit_subset: int = 100,
                        mirroring_strategy: Literal['pauli_rc', 'central_pauli'] = 'pauli_rc',
                        gate_set: str = 'u3_cx',
                        inverse: Optional[Callable[[_Circuit], _Circuit]] = None,
                        inv_kwargs: Optional[dict] = None,
                        rc_function: Optional[Callable[[_Circuit], Tuple[_Circuit, str]]] = None,
                        rc_kwargs: Optional[dict] = None,
                        cp_function: Optional[Callable[[_Circuit], Tuple[_Circuit, str]]] = None,
                        cp_kwargs: Optional[dict] = None,
                        state_initialization: Optional[Union[str, Callable[..., _Circuit]]] = None,
                        state_init_kwargs: Optional[dict] = None,
                        rand_state: Optional[_np.random.RandomState] = None) -> _CombinedExperimentDesign:
    # idea for random_compiling and state_initialization is that a power user could supply their own functions if they did not want to use the built-in options.

    # Please read carefully for guidelines on how design your own functions that implement a circuit inverse, a state prep, an RC, and a central Pauli propagation.
    # You may, but are not required to, provide custom implementations for supported gate sets.
    # For unsupported gate sets, you *must* provide a custom implementation.

    # Inverse method
    #   Signature: inverse(circ: pygsti.circuits.Circuit, ...) -> pygsti.circuits.Circuit
    #   You *must* have 'circ' as the circuit parameter name.
    #   Supply any additional kwargs needed by your custom inverse in the inv_kwargs kwarg when calling make_mirror_edesign().

    # State prep method
    #   Signature: state_initialization(qubits, rand_state: Optional[_np.random.RandomState] = None, ...) -> pygsti.circuits.Circuit
    #   You *must* have 'qubits' (list of the qubits being used) and 'rand_state' as parameter names.
    #   Supply any additional kwargs needed by your custom state initialization in the state_init_kwargs kwarg when calling make_mirror_design().

    # RC method
    #   Signature: rc_function(circ: pygsti.circuits.Circuit, rand_state: Optional[_np.random.RandomState] = None, ...) -> Tuple[pygsti.circuits.Circuit, str]
    #   The user-defined function must return the randomized circuit, along with the expected bitstring measurement given the randomization.
    #   This function is called twice:
    #       1) On the reverse half of the circuit, when creating the init-test-ref_inv-init_inv circuit. ref_inv and init_inv are randomized. The bitstring should be the expected measurement *for the full circuit*, *not* ref_inv-init_inv in isolation from the forward half of the circuit. Pass the forward half of the circuit as a kwarg in rc_kwargs if necessary.
    #       2) On the entire circuit, when creating the init-ref-ref_inv-init_inv circuit. All four pieces are randomized. The bitstring should again be the expected measurement for the full circuit, but it is more clear in this case than in the previous.
    #   Supply any additional kwargs needed by your custom RC function in the rc_kwargs kwarg when calling make_mirror_design().

    # CP method
    #   Signature: cp_function(forward_circ: pygsti.circuits.Circuit, reverse_circ: pygsti.circuits.Circuit, rand_state: Optional[_np.random.RandomState] = None, ...) -> Tuple[pygsti.circuits.Circuit, str]
    #   The user-defined function must return the CP circuit, along with the expected bitstring measurement given the central Pauli. The general responsibility can be thought of as creating and then "propagating" a random central Pauli layer.
    #   This function is called once, with init-test passed as the forward circuit and ref_inv-init_inv passed as the reverse circuit.
    #   Supply any additional kwargs needed by your custom CP function in the cp_kwargs kwarg when calling make_mirror_design().



    if rand_state is None:
        rand_state = _np.random.RandomState()

    qubit_subsets = defaultdict(list)

    test_ref_invs = defaultdict(list)
    ref_ref_invs = defaultdict(list)
    spam_refs = defaultdict(list)

    for c, auxlist in _tqdm.tqdm(test_edesign.aux_info.items(), ascii=True, desc='Sampling mirror circuits'):
        aux = auxlist[0]
        # print(f"auxlist length: {len(auxlist)}")
        # add a qubit subset to the pool if there's a circuit that uses it. if multiple identical circuits use it, do not double count; if distinct circuits use the same qubit subset, then do add it again.
        qubits = c.line_labels
        #print(qubits)

        qubit_subsets[aux['width']].append(qubits)

        if ref_edesign is not None:
            # find the corresponding exact circuit for the inexact circuit 'c' using the look-up table generated
            circ_id = aux['id']

            assert id_exact_circ_dict is not None, "when providing separate test and reference compilations, you must provide a lookup dictionary for the reference circuits so they can be matched with the correct test circuits."
            
            exact_circ = id_exact_circ_dict[circ_id]

            valid_test_ids = set([aux['id'] for aux in ref_edesign.aux_info[exact_circ]])

            test_id = aux['id']

            assert test_id in valid_test_ids, f"Invalid test ID {test_id} for ref circuit corresponding to test IDs {valid_test_ids}"

        else:
            print("using provided edesign for both reference and test compilations")
            exact_circ = c

        # R for "Reference" circuit, T for "Test" circuit
        R = exact_circ
        T = c

        R_inv = compute_inverse(circ=R, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)

        # print(R)
        # print(T)
        for j in range(num_mcs_per_circ):
            L = init_layer(qubits=qubits, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
                
            # compute needed circuit inverses, which are the SPAM layer and reference circuit inverse            
            L_inv = compute_inverse(circ=L, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)

            random_compiler = _rc.RandomCompilation(mirroring_strategy, rand_state)
            
            if mirroring_strategy == 'pauli_rc':
                if rc_function is not None:
                    try:
                        Rinv_Linv, L_T_Rinv_Linv_bs = rc_function(circ=R_inv + L_inv, rand_state=rand_state, **rc_kwargs)
                        L_T_Rinv_Linv = L + T + Rinv_Linv
                        L_R_Rinv_Linv, L_R_Rinv_Linv_bs = rc_function(circ=L + R + R_inv + L_inv, rand_state=rand_state, **rc_kwargs)
                    except:
                        raise RuntimeError(f"User-provided RC function for gate set '{gate_set}' returned an error!")
                elif gate_set == 'u3_cx':
                    Rinv_Linv, L_T_Rinv_Linv_bs = random_compiler.compile(R_inv + L_inv)
                    L_T_Rinv_Linv = L + T + Rinv_Linv
                    L_T_Rinv_Linv = L_T_Rinv_Linv.reorder_lines(c.line_labels)
                    L_R_Rinv_Linv, L_R_Rinv_Linv_bs = random_compiler.compile(L + R + R_inv + L_inv)
                elif gate_set == 'clifford':
                    #TODO: add clifford RC function
                    raise NotImplementedError("Clifford RC is not yet supported!")
                elif gate_set == 'clifford_rz':
                    #TODO: add clifford_rz RC function
                    raise NotImplementedError("Clifford_rz RC is not yet supported!")
                else: #we have no support for this gate set at this point, the user must provide a custom RC function
                    raise RuntimeError(f"No default RC function for gate set '{gate_set}' exists, you must provide your own!")

            elif mirroring_strategy == 'central_pauli':
                #do central pauli mirror circuit
                if cp_function is not None:
                    try:
                        L_T_Rinv_Linv, L_T_Rinv_Linv_bs = cp_function(forward_circ=L+T, reverse_circ=R_inv+L_inv, rand_state=rand_state, **cp_kwargs)
                    except:
                        raise RuntimeError(f"User-provided CP function for gate set '{gate_set}' returned an error!")
                elif gate_set == 'u3_cx':
                    #print(f'ref depth: {R.depth}, ref_inv depth: {R_inv.depth}, L depth: {L_inv.depth}')
                    #reload(cp)
                    # test_rand_state1 = _np.random.RandomState(8675309)

                    CP_Rinv_Linv, L_T_Rinv_Linv_bs = random_compiler.compile(circ=R_inv+L_inv)
                    L_T_Rinv_Linv = L + T + CP_Rinv_Linv
                    L_T_Rinv_Linv = L_T_Rinv_Linv.reorder_lines(c.line_labels)
                    # print("new function:")
                    # print(L_T_Rinv_Linv)
                    # print(L_T_Rinv_Linv_bs)
                    # test_rand_state2 = _np.random.RandomState(8675309)
                    # L_T_Rinv_Linv1, L_T_Rinv_Linv_bs1 = cp.central_pauli_mirror_circuit(test_circ=L+T, ref_circ=L+R, randomized_state_preparation=False, rand_state=test_rand_state2, new=True)
                    # print("old function:")
                    # print(L_T_Rinv_Linv1)
                    # print(L_T_Rinv_Linv_bs1)
                    # assert L_T_Rinv_Linv_bs == L_T_Rinv_Linv_bs1, "different circuit outcomes predicted!"
                    # assert L_T_Rinv_Linv.__hash__() == L_T_Rinv_Linv1.__hash__(), f"circuit hashes do not match! Circuit 1: {L_T_Rinv_Linv}, Circuit 2: {L_T_Rinv_Linv1}"

                    # L_R_Rinv_Linv, L_R_Rinv_Linv_bs = rc.central_pauli_mirror_circuit(test_circ=L+R, ref_circ=L+R, randomized_state_preparation=False, rand_state=rand_state, new=True)
                elif gate_set == 'clifford':
                    #TODO: add clifford CP function
                    raise NotImplementedError("Clifford CP is not yet supported!")
                elif gate_set == 'clifford_rz':
                    #TODO: add clifford_rz CP function
                    raise NotImplementedError("Clifford_rz CP is not yet supported!")
                else: #we have no support for this gate set at this point, the user must provide a custom CP function
                    raise RuntimeError(f"No default CP function for gate set '{gate_set}' exists, you must provide your own!")
                
            else:
                raise RuntimeError("'mirroring_strategy' must be either 'pauli_rc' or 'central_pauli'")
            
            L_T_Rinv_Linv_aux = [{'base_aux': a, 'idealout': L_T_Rinv_Linv_bs, 'id': j} for a in auxlist]
            test_ref_invs[L_T_Rinv_Linv] = L_T_Rinv_Linv_aux

            if mirroring_strategy == 'pauli_rc': #we do not have this class of circuit for central pauli
                L_R_Rinv_Linv_aux = [{'base_aux': a, 'idealout': L_R_Rinv_Linv_bs, 'id': j} for a in auxlist]
                ref_ref_invs[L_R_Rinv_Linv] = L_R_Rinv_Linv_aux
 
            # could change this to the same structure as described in the comment above if defaultdict(list) is not used

    for w, width_subsets in qubit_subsets.items():
        # subset_indices = rand_state.choice(list(range(len(width_subsets))), num_ref_per_qubit_subset)
        # for j in tqdm.tqdm(range(num_ref_per_qubit_subset), ascii=True, desc=f'Sampling width {w} reference circuits'):
        #     subset_idx = subset_indices[j]
        #     spam_qubits = width_subsets[subset_idx]
        #     L = init_layer(qubits=spam_qubits, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
        #     spam_ref = L + compute_inverse(circ=L, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)
        #     spam_refs[spam_ref].append({'idealout': '0'*w, 'id': j, 'width': w})
        #     # same defaultdict(list) change can be made as earlier
        
        unique_subsets = set(width_subsets)
        print(f'Sampling reference circuits for width {w} with {len(unique_subsets)} subsets')
        for unique_subset in unique_subsets:
            for j in _tqdm.tqdm(range(num_ref_per_qubit_subset), ascii=True, desc=f'Sampling reference circuits for subset {unique_subset}'):
                L = init_layer(qubits=unique_subset, gate_set=gate_set, state_initialization=state_initialization, rand_state=rand_state, state_init_kwargs=state_init_kwargs)
                spam_ref = L + compute_inverse(circ=L, gate_set=gate_set, inverse=inverse, inv_kwargs=inv_kwargs)
                spam_refs[spam_ref].append({'idealout': '0'*w, 'id': j, 'width': w})


    edesigns = {}
    if mirroring_strategy == 'pauli_rc':
        edesigns['br'] = _FreeformDesign(test_ref_invs)
        edesigns['rr'] = _FreeformDesign(ref_ref_invs)
        edesigns['ref'] = _FreeformDesign(spam_refs)
    elif mirroring_strategy == 'central_pauli':
        edesigns['cp'] = _FreeformDesign(test_ref_invs)
        edesigns['cpref'] = _FreeformDesign(spam_refs)
    else:
        raise RuntimeError("unknown mirroring strategy!")

    return _CombinedExperimentDesign(edesigns)


def compute_inverse(circ: _Circuit,
                    gate_set: str,
                    inverse: Optional[Callable[[_Circuit], _Circuit]] = None,
                    inv_kwargs: Optional[dict] = None) -> _Circuit:
    if inverse is not None:
        try:
            circ_inv = inverse(circ=circ, **inv_kwargs)
        except:
            raise RuntimeError(f"User-provided inverse function for gate set '{gate_set}' returned an error!")
    if gate_set == 'u3_cx':
        circ_inv = _rc.u3_cx_inv(circ)
    elif gate_set == 'clifford':
        #TODO: add clifford inverse function to circuit_inverse.py
        raise NotImplementedError("Clifford inversion is not yet supported!")
    elif gate_set == 'clifford_rz':
        #TODO: add clifford_rz inverse function to circuit_inverse.py
        raise NotImplementedError("Clifford_rz inversion is not yet supported!")
    else: #we have no support for this gate set at this point, the user must provide a custom inverse function
        raise RuntimeError(f"No default inverse function for gate set '{gate_set}' exists, you must provide your own!")
    
    return circ_inv

def init_layer(qubits,
               gate_set: str,
               state_initialization: Optional[Union[str, Callable[..., _Circuit]]] = None,
               rand_state: Optional[_np.random.RandomState] = None,
               state_init_kwargs: Optional[dict] = None) -> _Circuit:
    if state_initialization == 'none':
        L = _Circuit([], qubits)
    elif state_initialization is not None:
        try:
            L = state_initialization(qubits=qubits, rand_state=rand_state, **state_init_kwargs)
        except:
            raise RuntimeError(f"User-provided state_initialization function for gate set '{gate_set}' returned an error!")
    elif gate_set == 'u3_cx':
        prep_layer = _rc.haar_random_u3_layer(qubits, rand_state)
        L = _Circuit([prep_layer], qubits)
    elif gate_set == 'clifford':
        #TODO: add clifford default initialization
        raise NotImplementedError("Clifford state initialization is not yet supported!")
    elif gate_set == 'clifford_rz':
        #TODO: add clifford_rz default initialization
        raise NotImplementedError("Clifford_rz state initialization is not yet supported!")
    else: #we have no support for this gate set at this point, the user must provide a custom state initialization
        raise RuntimeError(f"No default state_initialization function for gate set '{gate_set}' exists, you must provide your own!")
    
    return L