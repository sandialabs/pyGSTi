# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
""" Core Idle Tomography routines """

import collections as _collections
import itertools as _itertools
import time as _time
import warnings as _warnings

import numpy as _np

from . import idttools as _idttools
from . import pauliobjs as _pobjs
from .idtresults import IdleTomographyResults as _IdleTomographyResults
from ... import baseobjs as _baseobjs
from ... import models as _models
from ... import tools as _tools
from ...models import modelconstruction as _modelconstruction
from ...modelmembers import states as _state
from ...modelmembers import operations as _op
from ...baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ...circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs import Basis
from itertools import product, permutations
from ...baseobjs import basisconstructors
import stim


# This module implements idle tomography, which deals only with
# many-qubit idle gates (on some number of qubits) and single-
# qubit gates (or tensor products of them) used to fiducials.
# As such, it is conventient to represent operations as native
# Python strings, where there is one I,X,Y, or Z letter per
# qubit.


# Commutator Helper Functions
def commute(mat1, mat2):
    return mat1 @ mat2 + mat2 @ mat1


def anti_commute(mat1, mat2):
    return mat1 @ mat2 - mat2 @ mat1


# NOTES ON THIS FACTOR OF TWO:
# IF/WHEN IT REEMERGES WITH ADDITIONAL QUBITS AND/OR
# HIGHER DIMENSION PAULIS:
# MAKE SURE THE INPUTS TO THESE FUNCTIONS ARE THE FULL PROCESS MATRICES
# AND NOT THE "ABBREVIATED" PAULIS


# Hamiltonian Error Generator
def hamiltonian_error_generator(initial_state, indexed_pauli, identity):
    return 2 * (
        -1j * indexed_pauli @ initial_state @ identity
        + 1j * identity @ initial_state @ indexed_pauli
    )


# Stochastic Error Generator
def stochastic_error_generator(initial_state, indexed_pauli, identity):
    return 2 * (
        indexed_pauli @ initial_state @ indexed_pauli
        - identity @ initial_state @ identity
    )


# Pauli-correlation Error Generator
def pauli_correlation_error_generator(
    initial_state,
    pauli_index_1,
    pauli_index_2,
):
    return 2 * (
        pauli_index_1 @ initial_state @ pauli_index_2
        + pauli_index_2 @ initial_state @ pauli_index_1
        - 0.5 * commute(commute(pauli_index_1, pauli_index_2), initial_state)
    )


# Anti-symmetric Error Generator
def anti_symmetric_error_generator(initial_state, pauli_index_1, pauli_index_2):
    return 2j * (
        pauli_index_1 @ initial_state @ pauli_index_2
        - pauli_index_2 @ initial_state @ pauli_index_1
        + 0.5
        * commute(
            anti_commute(pauli_index_1, pauli_index_2),
            initial_state,
        )
    )


## TODO: Update this function to use
## #pauli_matrices = basisconstructors.pp_matrices_dict(2**numQubits, normalize=False)
## as is now done elsewhere
# Convert basis
def convert_to_pauli(matrix, numQubits):
    # pauli_matrices = basisconstructors.pp_matrices_dict(2**numQubits, normalize=False)
    pauliNames1Q = ["I", "+X", "+Y", "+Z"]
    # Hard force to 1- or 2-qubit
    if numQubits == 1:
        pauliNames = pauliNames1Q
    elif numQubits == 2:
        pauliNames = ["".join(name) for name in product(pauliNames1Q, pauliNames1Q)]
    elif numQubits == 3:
        pauliNames = [
            "".join(name) for name in product(pauliNames1Q, pauliNames1Q, pauliNames1Q)
        ]
    pp = Basis.cast("PP", dim=4**numQubits)
    translationMatrix = pp.from_std_transform_matrix
    coefs = _np.real_if_close(_np.dot(translationMatrix, matrix.flatten()))
    return [(a, b) for (a, b) in zip(coefs, pauliNames) if abs(a) > 0.0001]

def convert_to_pauli_by_trace(matrix, numQubits):
    pp = Basis.cast("PP", dim=4**numQubits)
    # print(f"{pp.elements = }")
    # print(f"{pp.labels = }")
    # print(f"{pp.ellookup = }")
    coef_list = []
    for lbl, elt in pp.ellookup.items():
        coef_list.append((_np.real_if_close(_np.trace(elt.conj().T @ matrix)), lbl))
    x = [True for coef in coef_list if abs(coef[0]) > 0.0001]
    if len(x) > 1:
        raise ValueError
    return coef_list


# Compute the set of measurable effects of Hamiltonian error generators operating on two qubits in each of the specified eigenstates
# Return: hamiltonian error and coef dictionary
def gather_hamiltonian_jacobian_coefs(pauliDict, numQubits, printFlag=True):
    parities = list(product(["+1", "-1"], repeat=numQubits))
    hamiltonianErrorOutputs = dict()
    identKey = "I" * numQubits
    ident = pauliDict[identKey]

    pauliDictProduct = list(
        _itertools.product(
            [key for key in pauliDict.keys() if "I" not in key],
            [key for key in pauliDict.keys() if key != identKey],
        )
    )
    # print(f"{pauliDict = }")
    # print(f"{pauliDictProduct = }")
    
    paulis1Q = basisconstructors.pp_matrices_dict(2, normalize=False)
    for pauliPair in pauliDictProduct:
        for parity in parities:
            inputStateName = "".join(
                [parity[i][:-1] + pauliPair[0][i] for i in range(len(pauliPair[0]))]
            )
            # print(f"{inputStateName =}")
            inputStateItems = [
                int(parity[i]) * paulis1Q[pauliPair[0][i]]
                for i in range(len(pauliPair[0]))
            ]
            # print(f"{inputStateItems = }")
            if len(inputStateItems) == 1:
                inputState = inputStateItems[0]
            else:
                while len(inputStateItems) >= 2:
                    inputState = _np.kron(inputStateItems[0], inputStateItems[1])
                    inputStateItems[1] = inputState
                    inputStateItems.pop(0)
            # print(f"{inputState = }")
            indexedPauli = pauliDict[pauliPair[1]]
            # print(f"{pauliPair[1] = }")
            # print(f"{indexedPauli = }")
            inputState = ident / (2**numQubits) + inputState / (2**numQubits)

            process_matrix = hamiltonian_error_generator(
                inputState, indexedPauli, ident
            )
            # print(f"{process_matrix = }")
            decomposition = convert_to_pauli(process_matrix, numQubits)
            # flag =False
            # if len(decomposition) > 0:
            #     print(decomposition)
            #     flag = True
            # # decomposition = convert_to_pauli_by_trace(process_matrix,numQubits)
            # if flag:
            #     print(decomposition)
            #     quit()
            # if str(inputStateName) == "+Z+X":
            #     if pauliPair[1] == "XX":
            #         print(f"{decomposition = }")
            #         quit()
            # Stores elements as ((prep, measure), error gen index): non-zero coefficient

            for element in decomposition:
                # print(f"{inputStateName = }")
                # print(f"{element[1] = }")
                # print(f"{pauliPair[1]}")
                if "I" not in element[1]:
                    hamiltonianErrorOutputs[
                        ((str(inputStateName), element[1]), pauliPair[1])
                    ] = element[0]
            for element in decomposition:
                if "I" in element[1]:
                    print(element)
                    new_strings = []
                    if element[1].index("I") == 0:
                        new_strings.append("+X" + element[1][1:])
                        new_strings.append("+Y" + element[1][1:])
                        new_strings.append("+Z" + element[1][1:])
                    else:
                        new_strings.append(element[1][0:2] + "+X")
                        new_strings.append(element[1][0:2] + "+Y")
                        new_strings.append(element[1][0:2] + "+Z")
                    for stringl in new_strings:
                        hamiltonianErrorOutputs[((str(inputStateName), stringl), pauliPair[1])] = element[0]

                        
    # printFlag = True
    if printFlag:
        for key in hamiltonianErrorOutputs:
            print(key, "\n", hamiltonianErrorOutputs[key])
    # quit()
    return hamiltonianErrorOutputs


# Compute the set of measurable effects of stochastic error generators operating on a single qubit in each of the specified eigenstates
def gather_stochastic_jacobian_coefs(pauliDict, numQubits, printFlag=False):
    parities = list(product(["+1", "-1"], repeat=numQubits))
    stochasticErrorOutputs = dict()
    identKey = "I" * numQubits
    ident = pauliDict[identKey]

    pauliDictProduct = list(
        _itertools.product(
            [key for key in pauliDict.keys() if "I" not in key],
            [key for key in pauliDict.keys() if key != identKey],
        )
    )
    paulis1Q = basisconstructors.pp_matrices_dict(2, normalize=False)
    for pauliPair in pauliDictProduct:
        for parity in parities:
            inputStateName = "".join(
                [parity[i][:-1] + pauliPair[0][i] for i in range(len(pauliPair[0]))]
            )
            inputStateItems = [
                int(parity[i]) * paulis1Q[pauliPair[0][i]]
                for i in range(len(pauliPair[0]))
            ]
            if len(inputStateItems) == 1:
                inputState = inputStateItems[0]
            else:
                while len(inputStateItems) >= 2:
                    inputState = _np.kron(inputStateItems[0], inputStateItems[1])
                    inputStateItems[1] = inputState
                    inputStateItems.pop(0)
            indexedPauli = pauliDict[pauliPair[1]]
            inputState = ident / 2 + inputState / 2

            process_matrix = stochastic_error_generator(inputState, indexedPauli, ident)

            decomposition = convert_to_pauli(process_matrix, numQubits)
            for element in decomposition:
                if "I" not in element[1]:
                    stochasticErrorOutputs[
                        ((str(inputStateName), element[1]), pauliPair[1])
                    ] = element[0]
    if printFlag:
        for key in stochasticErrorOutputs:
            print(key, "\n", stochasticErrorOutputs[key])

    return stochasticErrorOutputs


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
def gather_pauli_correlation_jacobian_coefs(pauliDict, numQubits, printFlag=False):
    pauliCorrelationErrorOutputs = dict()
    parities = list(product(["+1", "-1"], repeat=numQubits))
    identKey = "I" * numQubits
    ident = pauliDict[identKey]

    pauliDictProduct = list(
        _itertools.permutations([key for key in pauliDict.keys() if key != identKey], 2)
    )

    pauliDictProduct = list(
        _itertools.product(
            [key for key in pauliDict.keys() if "I" not in key], pauliDictProduct
        )
    )
    paulis1Q = basisconstructors.pp_matrices_dict(2, normalize=False)
    for pauliPair in pauliDictProduct:
        for parity in parities:
            inputStateName = "".join(
                [parity[i][:-1] + pauliPair[0][i] for i in range(len(pauliPair[0]))]
            )
            inputStateItems = [
                int(parity[i]) * paulis1Q[pauliPair[0][i]]
                for i in range(len(pauliPair[0]))
            ]
            if len(inputStateItems) == 1:
                inputState = inputStateItems[0]
            else:
                while len(inputStateItems) >= 2:
                    inputState = _np.kron(inputStateItems[0], inputStateItems[1])
                    inputStateItems[1] = inputState
                    inputStateItems.pop(0)
            indexedPauli1 = pauliDict[pauliPair[1][0]]
            indexedPauli2 = pauliDict[pauliPair[1][1]]
            inputState = ident / 2 + inputState / 2
            process_matrix = pauli_correlation_error_generator(
                inputState, indexedPauli1, indexedPauli2
            )
            decomposition = convert_to_pauli(process_matrix, numQubits)
            for element in decomposition:
                if "I" not in element[1]:
                    pauliCorrelationErrorOutputs[
                        ((str(inputStateName), element[1]), pauliPair[1])
                    ] = element[0]

    if printFlag:
        for key in pauliCorrelationErrorOutputs:
            print(key, "\n", pauliCorrelationErrorOutputs[key])

    return pauliCorrelationErrorOutputs


# Compute the set of measurable effects of pauli-correlation error generators operating on a single qubit in each of the specified eigenstates
def gather_anti_symmetric_jacobian_coefs(pauliDict, numQubits, printFlag=False):
    antiSymmetricErrorOutputs = dict()
    parities = list(product(["+1", "-1"], repeat=numQubits))
    identKey = "I" * numQubits
    ident = pauliDict[identKey]

    pauliDictProduct = list(
        _itertools.permutations([key for key in pauliDict.keys() if key != identKey], 2)
    )

    pauliDictProduct = list(
        _itertools.product(
            [key for key in pauliDict.keys() if "I" not in key], pauliDictProduct
        )
    )
    paulis1Q = basisconstructors.pp_matrices_dict(2, normalize=False)
    for pauliPair in pauliDictProduct:
        for parity in parities:
            inputStateName = "".join(
                [
                    str(parity[i])[:-1] + pauliPair[0][i]
                    for i in range(len(pauliPair[0]))
                ]
            )
            inputStateItems = [
                int(parity[i]) * paulis1Q[pauliPair[0][i]]
                for i in range(len(pauliPair[0]))
            ]
            if len(inputStateItems) == 1:
                inputState = inputStateItems[0]
            else:
                while len(inputStateItems) >= 2:
                    inputState = _np.kron(inputStateItems[0], inputStateItems[1])
                    inputStateItems[1] = inputState
                    inputStateItems.pop(0)
            indexedPauli1 = pauliDict[pauliPair[1][0]]
            indexedPauli2 = pauliDict[pauliPair[1][1]]
            inputState = ident / 2 + inputState / 2
            process_matrix = anti_symmetric_error_generator(
                inputState, indexedPauli1, indexedPauli2
            )
            decomposition = convert_to_pauli(process_matrix, numQubits)
            for element in decomposition:
                if "I" not in element[1]:
                    antiSymmetricErrorOutputs[
                        ((str(inputStateName), element[1]), pauliPair[1])
                    ] = element[0]
    if printFlag:
        for key in antiSymmetricErrorOutputs:
            print(key, "\n", antiSymmetricErrorOutputs[key])

    return antiSymmetricErrorOutputs


def build_class_jacobian(classification, numQubits):
    pauli_matrices = basisconstructors.pp_matrices_dict(2**numQubits, normalize=False)
    print(len(pauli_matrices))
    identKey = "I" * numQubits

    # classification within ["H", "S", "C", "A"]
    if classification == "H":
        jacobian_coefs = gather_hamiltonian_jacobian_coefs(
            pauliDict=pauli_matrices,
            numQubits=numQubits,
            printFlag=False,
        )
        # print(jacobian_coefs)

    elif classification == "S":
        jacobian_coefs = gather_stochastic_jacobian_coefs(
            pauliDict=pauli_matrices,
            numQubits=numQubits,
            printFlag=False,
        )
        # print(jacobian_coefs)
    elif classification == "C":
        jacobian_coefs = gather_pauli_correlation_jacobian_coefs(
            pauliDict=pauli_matrices,
            numQubits=numQubits,
            printFlag=False,
        )
        # print(jacobian_coefs)
    elif classification == "A":
        jacobian_coefs = gather_anti_symmetric_jacobian_coefs(
            pauliDict=pauli_matrices,
            numQubits=numQubits,
            printFlag=False,
        )
        # print(jacobian_coefs)
    else:
        print(
            "Classification value must be 'H', 'S', 'C', or 'A'.  Please provide a valid argument."
        )
        quit()

    return jacobian_coefs


def dict_to_jacobian(coef_dict, classification, numQubits, all_fidpairs):
    pauliDict = basisconstructors.pp_matrices_dict(2**numQubits, normalize=False)
    identKey = "I" * numQubits
    parities = list(product([1, -1], repeat=numQubits))
    paulis1Q = basisconstructors.pp_matrices_dict(2, normalize=False)
    if classification == "H" or classification == "S":
        pauliDictProduct = dict(
            enumerate([key for key in pauliDict.keys() if key != identKey])
        )

    elif classification == "C" or classification == "A":
        # NOTE: this may actually be combinations to avoid double counting/symmetry
        pauliDictProduct = dict(
            enumerate(
                list(
                    _itertools.combinations(
                        [key for key in pauliDict.keys() if key != identKey], 2
                    )
                )
            )
        )

        # print(f"{pauliDictProduct = }")
        # quit()

    else:
        print(
            "Classification value must be 'H', 'S', 'C', or 'A'.  Please provide a valid argument."
        )
        quit()
    row_index_list = {(str(v[0]), str(v[1])): k for (k, v) in all_fidpairs.items()}

    col_index_list = {v: k for (k, v) in pauliDictProduct.items()}
    # print(col_index_list)
    output_jacobian = _np.zeros((len(all_fidpairs), len(pauliDictProduct)))
    # print(f"{row_index_list = }")
    # print(f"{col_index_list = }")
    # print(f"{coef_dict = }")
    # quit()
    for coef in coef_dict:
        row_index_coef = row_index_list.get(coef[0])
        col_index_coef = col_index_list.get(coef[1])
        if row_index_coef:
            # print(f"{coef = }")
            output_jacobian[row_index_coef][col_index_coef] = coef_dict[coef]
    # print(output_jacobian)
    # print(col_index_list)
    return output_jacobian, col_index_list


#--------- Test out Hamiltonian Jacobian Stuff ---------------#

def hamiltonian_jac_alt(num_qubits, prep_meas_fid_pairs = None):
    #The jacobian we are trying to construct has columns corresponding to
    #intrisic hamiltonian generator rates, and rows corresponding to prep-measure
    #experiment pairs, where prep means prepare some eigenstate of some pauli,
    #and measure means measure the inner product with some eigenstate of some
    #pauli. We can write these as (I+Q)/sqrt(d) and (I+R)/sqrt(d) for some
    #signed paulis Q and R.

    #Dictionary of all of the n-qubit (unsigned) paulis.
    pauli_matrices = basisconstructors.pp_matrices_dict(2**num_qubits, normalize=False)

    #all possible parities for the pauli strings
    parities = list(product([1, -1], repeat=num_qubits))

    #get rid of the all identity string, we don't need it.
    identKey = "I" * num_qubits
    del pauli_matrices[identKey]

    #constuct list of strictly non-trivial paulis for prep and measurement
    all_nontrivial_paulis = [key for key in pauli_matrices.keys() if "I" not in key]
    
    #for ham keys the only restriction is not the all identity string
    #let's make the hamiltonian_pauli_idxs values stim PauliString objects
    hamiltonian_pauli_idxs = [stim.PauliString(pauli) for pauli in pauli_matrices.keys()]
    #if prep_meas_eigenstate_pairs are specified then use these,
    #casting to appropriate overall signed stim PauliStrings
    #otherwise, generate a set of all possible pairs, but with
    #the last all '-' parity skipped on the measurements (to get
    #the number of dof correct).

    if prep_meas_fid_pairs is None:
        #For the eigenstates, we are going to want to loop through the non-trivial paulis
        #and construct the different possible parities.
        signed_nontrivial_paulis_prep = []
        signed_nontrivial_paulis_meas = []
        
        #an alternate version of the signed paulis where we only track
        #the overall sign from pulling the scalars signs out front.
        overall_signed_nontrivial_paulis_prep = []
        overall_signed_nontrivial_paulis_meas = []

        for pauli in all_nontrivial_paulis:
            for j, parity in enumerate(parities):
                signed_pauli = ''
                for i,subpauli in enumerate(pauli):
                    if parity[i]>0:
                        signed_pauli += f'+{subpauli}'
                    else:
                        signed_pauli += f'-{subpauli}'
                signed_nontrivial_paulis_prep.append(signed_pauli)
                if _np.prod(parity)>0:
                    overall_signed_pauli = '+' + pauli
                else:
                    overall_signed_pauli = '-' + pauli
                #append this as a stim PauliString object,
                #which can handle overall signed paulis (but not individually
                #signed ones).
                overall_signed_nontrivial_paulis_prep.append(stim.PauliString(overall_signed_pauli))
                
                #skip the all - parity on the meas
                if j != (len(parities)-1):
                    signed_nontrivial_paulis_meas.append(signed_pauli)
                    overall_signed_nontrivial_paulis_meas.append(stim.PauliString(overall_signed_pauli))
        #loop through  prep_meas_pairs, create a list of tuples corresponding to prep eigenstates
        #and meas eigenstates.
        prep_meas_eigenstate_pairs = list(product(signed_nontrivial_paulis_prep, signed_nontrivial_paulis_meas))
        #create a similar iterator but for the overall-signed pairs.
        prep_meas_overall_signed_pairs = list(product(overall_signed_nontrivial_paulis_prep, overall_signed_nontrivial_paulis_meas))
        #otherwise a set of prep measure pairs has been specified. For now assume that this has been specified
        #in the form of a list of tuples of NQPauliState objects, though we'll eventually probably loosen this.    
    else:
        prep_meas_eigenstate_pairs = []
        prep_meas_overall_signed_pairs = []
        for prep, meas in prep_meas_fid_pairs:
            prep_meas_eigenstate_pairs.append((str(prep), str(meas)))
            #remove the pluses and minuses
            stripped_prep = str(prep).replace('+', '').replace('-', '')
            stripped_meas = str(meas).replace('+', '').replace('-', '')
            
            #calculate the overall parity
            overall_sign_prep = _np.prod(prep.signs) #(-1)**_np.sum(prep.signs)
            overall_sign_meas = _np.prod(meas.signs) #(-1)**_np.sum(meas.signs)

            overall_signed_pauli_prep = '+' + stripped_prep if overall_sign_prep>0 else '-' + stripped_prep
            overall_signed_pauli_meas = '+' + stripped_meas if overall_sign_meas>0 else '-' + stripped_meas

            prep_meas_overall_signed_pairs.append((stim.PauliString(overall_signed_pauli_prep), stim.PauliString(overall_signed_pauli_meas)))

    #initialize an empty numpy array for the jacobian:
    ham_jac = _np.zeros((len(prep_meas_eigenstate_pairs), len(hamiltonian_pauli_idxs)))

    #for each of these pairs we then want to loop through all of the elements of
    #hamiltonian_pauli_idxs and compute the component of the jacobian for this
    #hamiltonian index.
    for row_idx, ((prep, meas), (prep_overall, meas_overall)) in enumerate(zip(prep_meas_eigenstate_pairs, prep_meas_overall_signed_pairs)):
        for col_idx, ham_pauli_idx in enumerate(hamiltonian_pauli_idxs):
            #we're calculating for each prep_overall, meas_overall, ham_pauli_idx
            #triple the expression (-i/d)*Tr(meas_overall@[ham_pauli_idx,prep_overall])
            prep_ham_idx_comm = half_pauli_comm(ham_pauli_idx, prep_overall)
            #if this commutator is zero then we can move onto the next hamiltonian index.
            if prep_ham_idx_comm == 0:
                continue

            #The paulis are trace orthonormal, so we only get a non-zero value
            #if meas_overall ~= prep_ham_idx_comm (i.e. up to the overall sign/phase).
            if is_unsigned_pauli_equiv(meas_overall, prep_ham_idx_comm):
                #in this case we just need to get the overall sign and append
                #an appropriately signed 2 to the jacobian.
                #this should be real-valued up to numerical precision,
                #but if not this will raise an error below. when we try
                #to set ham_jac which has a dtype of np.double. 
                final_sign = _np.real_if_close(-1j*meas_overall.sign*prep_ham_idx_comm.sign)
                ham_jac[row_idx, col_idx] = final_sign*2

    #TODO: Delete second return, this is currently only for testing/diagnostics.
    return ham_jac, prep_meas_eigenstate_pairs

def half_pauli_comm(pauli1, pauli2):
    '''
    Computes the commutator between two unsigned pauli strings.
    [pauli1, pauli2]

    Parameters
    ----------
    pauli1 : stim.PauliString
    
    pauli2 : stim.PauliString
    
    Returns
    -------
    comm : stim.PauliString or int
        The signed pauli string corresponding to the commutator,
        up to a missing factor of 2.
        If the paulis commute then 0 is returned.
    '''
    
    #Every pair of paulis either commute or anticommute (see 
    #https://arxiv.org/pdf/1909.08123.pdf for more on commutation structure
    #of the n-qubit pauli group). So this breaks into two cases. If
    #pauli1 and pauli2 commute, then [pauli1, pauli2] = 0. If they anticommute
    #then [pauli1,pauli2] = 2*pauli1@pauli2. Stim has a nice built-in method
    #for checking this. But, stim doesn't have a nice way to represent a scalar
    #multiple of a pauli string, so we'll just return the product.

    if pauli1.commutes(pauli2):
        return 0
    else:
        return pauli1*pauli2

def is_unsigned_pauli_equiv(pauli1, pauli2):
    '''
    Determines whether two paulis are equivalent up to overall phase.

    Parameters
    ----------
    pauli1 : stim.PauliString
    
    pauli2 : stim.PauliString
    
    Returns
    -------
    is_equiv : bool
        True of paulis are equivalent up to phase, otherwise
        False.
    '''

    #create a copy of the paulis, then set their sign attributes to be +1.
    #we can then compare these using stim's equivalence check.
    pauli1 = pauli1.copy()
    pauli2 = pauli2.copy()

    pauli1.sign = 1
    pauli2.sign = 1

    return pauli1 == pauli2








    



# -----------------------------------------------------------------------------
# Experiment generation:
# -----------------------------------------------------------------------------


# def idle_tomography_fidpairs(
#    nqubits,
#    maxweight=2,
#    include_hamiltonian=True,
#    include_stochastic=True,
#    include_correlation= True,
#    include_active=True,
#    ham_tmpl="auto",
#    preferred_prep_basis_signs=("+", "+", "+"),
#    preferred_meas_basis_signs=("+", "+", "+"),
# ):
#    """
#    Construct a list of Pauli-basis fiducial pairs for idle tomography.
#
#    This function constructs the "standard" set of fiducial pairs used
#    to generate idle tomography sequences which probe Hamiltonian,
#    Stochastic, and/or active errors in an idle gate.
#
#    Parameters
#    ----------
#    nqubits : int
#        The number of qubits.
#
#    maxweight : int, optional
#        The maximum weight of errors to consider.
#
#    include_hamiltonian, include_stochastic, include_active : bool, optional
#        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
#        and active-type errors.
#
#    ham_tmpl : tuple, optional
#        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
#        "Y", or "Z"), describing how to construct the fiducial pairs used to
#        detect Hamiltonian errors.  The special (and default) value "auto"
#        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
#        `maxweight` equal to 1 and 2, repectively, and will generate an error
#        if `maxweight > 2`.
#
#    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
#        A 3-tuple of "+" or "-" strings indicating which sign for preparing
#        or measuring in the X, Y, and Z bases is preferable.  Usually one
#        orientation if preferred because it's easier to achieve using the
#        native model.
#
#    Returns
#    -------
#    list
#        a list of (prep,meas) 2-tuples of NQPauliState objects, each of
#        length `nqubits`, representing the fiducial pairs.
#    """
#    fidpairs = []  # list of 2-tuples of NQPauliState objects to return
#
#    # convert +'s and -'s to dictionaries of +/-1 used later:
#    def conv(x):
#        return 1 if x == "+" else -1
#
#    base_prep_signs = {
#        l: conv(s) for l, s in zip(("X", "Y", "Z"), preferred_prep_basis_signs)
#    }
#    base_meas_signs = {
#        l: conv(s) for l, s in zip(("X", "Y", "Z"), preferred_meas_basis_signs)
#    }
#    # these dicts give the preferred sign for prepping or measuring along each 1Q axis.
#
#    if include_stochastic:
#        if include_active:
#            # in general there are 2^maxweight different permutations of +/- signs
#            # in maxweight==1 case, need 2 of 2 permutations
#            # in maxweight==2 case, need 3 of 4 permutations
#            # higher maxweight?
#
#            if maxweight == 1:
#                flips = [
#                    (1,),
#                    (-1,),
#                ]  # consider both cases of not-flipping & flipping the preferred basis signs
#
#            elif maxweight == 2:
#                flips = [
#                    (1, 1),  # don't flip anything
#                    (1, -1),
#                    (-1, 1),
#                ]  # flip 2nd or 1st pauli basis (weight = 2)
#            else:
#                raise NotImplementedError(
#                    "No implementation for active errors and maxweight > 2!"
#                )
#                # need to do more work to figure out how to generalize this to maxweight > 2
#        else:
#            flips = [(1,) * maxweight]  # don't flip anything
#
#        # Build up "template" of 2-tuples of NQPauliState objects acting on
#        # maxweight qubits that should be tiled to full fiducial pairs.
#        sto_tmpl_pairs = []
#        for fliptup in flips:  # elements of flips must have length=maxweight
#            # Create a set of "template" fiducial pairs using the current flips
#            for basisLets in _itertools.product(("X", "Y", "Z"), repeat=maxweight):
#                # flip base (preferred) basis signs as instructed by fliptup
#                prep_signs = [
#                    f * base_prep_signs[l] for f, l in zip(fliptup, basisLets)
#                ]
#                meas_signs = [
#                    f * base_meas_signs[l] for f, l in zip(fliptup, basisLets)
#                ]
#                sto_tmpl_pairs.append(
#                    (
#                        _pobjs.NQPauliState("".join(basisLets), prep_signs),
#                        _pobjs.NQPauliState("".join(basisLets), meas_signs),
#                    )
#                )
#
#        fidpairs.extend(
#            _idttools.tile_pauli_fidpairs(sto_tmpl_pairs, nqubits, maxweight)
#        )
#
#    elif include_active:
#        raise ValueError(
#            "Cannot include active sequences without also including stochastic ones!"
#        )
#
#    if include_hamiltonian:
#        nextPauli = {"X": "Y", "Y": "Z", "Z": "X"}
#        prevPauli = {"X": "Z", "Y": "X", "Z": "Y"}
#
#        def prev(expt):
#            return "".join([prevPauli[p] for p in expt])
#
#        def next(expt):
#            return "".join([nextPauli[p] for p in expt])
#
#        if ham_tmpl == "auto":
#            if maxweight == 1:
#                ham_tmpl = ("X", "Y", "Z")
#            elif maxweight == 2:
#                ham_tmpl = ("ZY", "ZX", "XZ", "YZ", "YX", "XY")
#            else:
#                raise ValueError("Must supply `ham_tmpl` when `maxweight > 2`!")
#        ham_tmpl_pairs = []
#        for tmplLets in ham_tmpl:  # "Lets" = "letters", i.e. 'X', 'Y', or 'Z'
#            assert len(tmplLets) == maxweight, (
#                "Hamiltonian 'template' strings must have length == maxweight: len(%s) != %d!"
#                % (tmplLets, maxweight)
#            )
#
#            prepLets, measLets = prev(tmplLets), next(tmplLets)
#
#            # basis sign doesn't matter for hamiltonian terms,
#            #  so just use preferred signs
#            prep_signs = [base_prep_signs[l] for l in prepLets]
#            meas_signs = [base_meas_signs[l] for l in measLets]
#            ham_tmpl_pairs.append(
#                (
#                    _pobjs.NQPauliState(prepLets, prep_signs),
#                    _pobjs.NQPauliState(measLets, meas_signs),
#                )
#            )
#
#        fidpairs.extend(
#            _idttools.tile_pauli_fidpairs(ham_tmpl_pairs, nqubits, maxweight)
#        )
#
#    return fidpairs


def idle_tomography_fidpairs(nqubits):
    """
    Construct a list of Pauli-basis fiducial pairs for idle tomography.

    This function simply does the most naive, symmetric experiment designations
    possible, which consists of all possible pairs of paulis, and all possible
    signs on the prep fiducial.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    Returns
    -------
    list
        a list of (prep,meas) 2-tuples of NQPauliState objects, each of
        length `nqubits`, representing the fiducial pairs.
    """

    pauli_strings = ["X", "Y", "Z"]
    nq_pauli_strings = list(
        product(pauli_strings, repeat=nqubits)
    )  # skip the all identity string]
    # nq_pauli_strings.pop(nq_pauli_strings.index("I"*nqubits))

    # we also want all possible combinations of sign for each the pauli
    # observable on each qubit. The NQPauliState expects these to be either 0
    # for + or 1 for -.
    signs = list(product([-1, 1], repeat=nqubits))

    fidpairs = []
    for prep_string, meas_string in product(nq_pauli_strings, repeat=2):
        for sign in signs:
            fidpairs.append(
                (
                    _pobjs.NQPauliState(prep_string, sign),
                    _pobjs.NQPauliState(meas_string, signs[-1]),
                )
            )

    return fidpairs


def preferred_signs_from_paulidict(pauli_basis_dict):
    """
    Infers what the preferred basis signs are based on the length of gate-name
    strings in `pauli_basis_dict` (shorter strings are preferred).

    Parameters
    ----------
    pauli_basis_dict : dict
        A dictionary w/keys like `"+X"` or `"-Y"` and values that
        are tuples of gate *names* (not labels, which include qubit or
        other state-space designations), e.g. `("Gx","Gx")`.

    Returns
    -------
    tuple
        A 3-tuple of elements in {"+", "-"}, exactly the format expected
        by `preferred_*_basis_signs` arguments of
        :function:`idle_tomography_fidpairs`.
    """
    preferred_signs = ()
    for let in ("X", "Y", "Z"):
        if "+" + let in pauli_basis_dict:
            plusKey = "+" + let
        elif let in pauli_basis_dict:
            plusKey = let
        else:
            plusKey = None

        if "-" + let in pauli_basis_dict:
            minusKey = "-" + let
        else:
            minusKey = None

        if minusKey and plusKey:
            if len(pauli_basis_dict[plusKey]) <= len(pauli_basis_dict[minusKey]):
                preferred_sign = "+"
            else:
                preferred_sign = "-"
        elif plusKey:
            preferred_sign = "+"
        elif minusKey:
            preferred_sign = "-"
        else:
            raise ValueError("No entry for %s-basis!" % let)

        preferred_signs += (preferred_sign,)

    return preferred_signs


def fidpairs_to_pauli_fidpairs(fidpairs_list, pauli_basis_dicts, nqubits):
    """
    Translate :class:`GatesString`-type fiducial pairs to
    :class:`NQPauliState`-type "Pauli fiducial pairs" using `pauli_basis_dicts`.

    Parameters
    ----------
    fidpairs_list : list
        A list whose elements are 2-tuples of :class:`Circuit` objects.

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    nqubits : int
        The number of qubits.  Needed because :class:`Circuit`
        objects don't contain this information.

    Returns
    -------
    list
        A list of 2-tuples of :class:`NQPauliState` objects.
    """

    # Example dicts:
    # prepDict = { 'X': ('Gy',), 'Y': ('Gx',)*3, 'Z': (),
    #         '-X': ('Gy',)*3, '-Y': ('Gx',), '-Z': ('Gx','Gx')}
    # measDict = { 'X': ('Gy',)*3, 'Y': ('Gx',), 'Z': (),
    #         '-X': ('Gy',), '-Y': ('Gx',)*3, '-Z': ('Gx','Gx')}
    prepDict, measDict = pauli_basis_dicts

    for k, v in prepDict.items():
        assert k[-1] in ("X", "Y", "Z") and isinstance(
            v, tuple
        ), "Invalid prep pauli dict format!"
    for k, v in measDict.items():
        assert k[-1] in ("X", "Y", "Z") and isinstance(
            v, tuple
        ), "Invalid measuse pauli dict format!"

    rev_prepDict = {v: k for k, v in prepDict.items()}
    rev_measDict = {v: k for k, v in measDict.items()}

    def convert(opstr, rev_pauli_dict):
        # Get gatenames_per_qubit (keys = sslbls, vals = lists of gatenames)
        # print("DB: Converting ",opstr)
        gatenames_per_qubit = _collections.defaultdict(list)
        for glbl in opstr:
            for c in glbl.components:  # in case of parallel labels
                assert len(c.sslbls) == 1
                assert isinstance(c.sslbls[0], int)
                gatenames_per_qubit[c.sslbls[0]].append(c.name)
        # print("DB: gatenames_per_qubit =  ",gatenames_per_qubit)
        # print("DB: rev keys = ",list(rev_pauli_dict.keys()))

        # Check if list of gatenames equals a known basis prep/meas:
        letters = ""
        signs = []
        for i in range(nqubits):
            basis = rev_pauli_dict.get(tuple(gatenames_per_qubit[i]), None)
            # print("DB:  Q%d: %s -> %s" % (i,str(gatenames_per_qubit[i]), str(basis)))
            assert basis is not None  # to indicate convert failed
            letters += basis[-1]  # last letter of basis should be 'X' 'Y' or 'Z'
            signs.append(-1 if (basis[0] == "-") else 1)

        # print("DB: SUCCESS: --> ",letters,signs)
        return _pobjs.NQPauliState(letters, signs)

    ret = []
    for prepStr, measStr in fidpairs_list:
        try:
            prepPauli = convert(prepStr, rev_prepDict)
            measPauli = convert(measStr, rev_measDict)
        except AssertionError:
            continue  # skip strings we can't convert
        ret.append((prepPauli, measPauli))

    return ret


def determine_paulidicts(model):
    """
    Intelligently determine preparation and measurement Pauli basis
    dictionaries from a :class:`Model`.

    The returned dictionaries are required for various parts of idle tomography,
    as they bridge the native model's gates to the "Pauli basis language"
    used in idle tomography.

    Parameters
    ----------
    model : Model
        The model which defines the available preparation, measurement, and
        operations.  It is assumed that `model`'s operation are expressed
        in a Pauli-product basis.

    Returns
    -------
    pauli_basis_dicts or None
        If successful, a `(prepDict,measureDict)` 2-tuple of Pauli basis
        dictionaries.  If unsuccessful, None.
    """
    # TODO: check that basis == "pp" or something similar?
    # Note: this routine just punts if model's operation labels are just strings.

    model._clean_paramvec()  # to ensure calls to obj.to_vector work below (setup model paramvec)

    # First, check that spam is prep/meas in Z basis (just check prep for now):
    try:
        prepLbls = list(model.preps.keys())
        prep = model.preps[
            prepLbls[0]
        ]  # just take the first one (usually there's only one anyway)
    except AttributeError:  # HACK to work w/Implicit models
        prepLbls = list(model.prep_blks["layers"].keys())
        prep = model.prep_blks["layers"][prepLbls[0]]

    if isinstance(prep, _state.ComputationalBasisState):
        if any([b != 0 for b in prep._zvals]):
            return None
    elif isinstance(prep, _state.ComposedState):
        if isinstance(prep.state_vec, _state.ComputationalBasisState):
            if any([b != 0 for b in prep.state_vec._zvals]):
                return None
        if any([abs(v) > 1e-6 for v in prep.to_vector()]):
            return None
    else:
        nqubits = int(round(_np.log2(model.dim) / 2))
        cmp = _state.ComputationalBasisState(
            [0] * nqubits, "pp", model._evotype
        ).to_dense()
        if _np.linalg.norm(prep.to_dense() - cmp) > 1e-6:
            return None

    def extract_action(g, cur_sslbls, ql):
        """Note: assumes cur_sslbs is just a list of labels (of first "sector"
        of a real StateSpaceLabels struct)"""
        if isinstance(g, _op.ComposedOp):
            action = _np.identity(4, "d")
            for fg in g.factorops:
                action = _np.dot(extract_action(fg, cur_sslbls, ql), action)
            return action

        if isinstance(g, _op.EmbeddedOp):
            # Note: an embedded gate need not use the *same* state space labels as the model
            g_sslbls = g.state_space.sole_tensor_product_block_labels
            lbls = [cur_sslbls[g_sslbls.index(locLbl)] for locLbl in g.target_labels]
            # TODO: add to StateSpaceLabels functionality to make sure two are compatible, and to translate between
            # them, & make sub-labels?
            return extract_action(g.embedded_op, lbls, ql)

        # StaticArbitraryOp, LindbladDenseOp, other gates...
        if len(cur_sslbls) == 1 and cur_sslbls[0] == ql:
            mx = g.to_dense()
            assert mx.shape == (4, 4)
            return mx
        else:
            mx = g.to_dense()
            if _np.linalg.norm(mx - _np.identity(g.dim, "d")) < 1e-6:
                # acts as identity on some other space - this is ok
                return _np.identity(4, "d")
            else:
                raise ValueError(
                    "LinearOperator acts nontrivially on a space other than that in its label!"
                )

    # Get several standard 1-qubit pi/2 rotations in Pauli basis:
    pp = _baseobjs.BuiltinBasis("pp", 4)
    Gx = _modelconstruction.create_operation(
        "X(pi/2,Q0)", [("Q0",)], basis=pp, parameterization="static"
    ).to_dense()
    Gy = _modelconstruction.create_operation(
        "Y(pi/2,Q0)", [("Q0",)], basis=pp, parameterization="static"
    ).to_dense()

    # try to find 1-qubit pi/2 rotations
    found = {}
    for gl in model.primitive_op_labels:
        if isinstance(model, _models.ExplicitOpModel):
            gate = model.operations[gl]
        else:
            gate = model.operation_blks["layers"][gl]

        if gl.sslbls is None or len(gl.sslbls) != 1:
            continue  # skip gates that don't have 1Q-like labels
        qubit_label = gl.sslbls[0]  # the qubit this gate is supposed to act on
        try:
            assert (
                model.state_space.num_tensor_product_blocks == 1
            ), "Assumes a single state space sector"
            action_on_qubit = extract_action(
                gate, model.state_space.sole_tensor_product_block_labels, qubit_label
            )
        except ValueError:
            continue  # skip gates that we can't extract action from

        # See if we recognize this action
        # FUTURE: add more options for using other existing gates?
        if _np.linalg.norm(action_on_qubit - Gx) < 1e-6:
            found["Gx"] = gl.name
        elif _np.linalg.norm(action_on_qubit - Gy) < 1e-6:
            found["Gy"] = gl.name

    if "Gx" in found and "Gy" in found:
        Gxl = found["Gx"]
        Gyl = found["Gy"]

        prepDict = {
            "X": (Gyl,),
            "Y": (Gxl,) * 3,
            "Z": (),
            "-X": (Gyl,) * 3,
            "-Y": (Gxl,),
            "-Z": (Gxl, Gxl),
        }
        measDict = {
            "X": (Gyl,) * 3,
            "Y": (Gxl,),
            "Z": (),
            "-X": (Gyl,),
            "-Y": (Gxl,) * 3,
            "-Z": (Gxl, Gxl),
        }
        return prepDict, measDict

    return None


def make_idle_tomography_list(
    nqubits,
    max_lengths,
    pauli_basis_dicts,
    maxweight=2,
    idle_string=((),),
    include_hamiltonian=True,
    include_stochastic=True,
    include_active=True,
    ham_tmpl="auto",
    preferred_prep_basis_signs="auto",
    preferred_meas_basis_signs="auto",
    force_fid_pairs=False,
):
    """
    Construct the list of experiments needed to perform idle tomography.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    max_lengths : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    include_hamiltonian, include_stochastic, include_active : bool, optional
        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
        and active-type errors.

    ham_tmpl : tuple, optional
        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
        "Y", or "Z"), describing how to construct the fiducial pairs used to
        detect Hamiltonian errors.  The special (and default) value "auto"
        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
        `maxweight` equal to 1 and 2, repectively, and will generate an error
        if `maxweight > 2`.

    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
        A 3-tuple of "+" or "-" strings indicating which sign for preparing
        or measuring in the X, Y, and Z bases is preferable.  Usually one
        orientation if preferred because it's easier to achieve using the
        native model.  Additionally, the special (and default) value "auto"
        may be used, in which case :function:`preferred_signs_from_paulidict`
        is used to choose preferred signs based on `pauli_basis_dicts`.

    Returns
    -------
    list
        A list of :class:`Circuit` objects.
    """

    prepDict, measDict = pauli_basis_dicts
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    GiStr = _Circuit(idle_string, num_lines=nqubits)
    if force_fid_pairs:
        pauli_fidpairs = force_fid_pairs
    else:
        pauli_fidpairs = idle_tomography_fidpairs(
            nqubits,
            maxweight,
            include_hamiltonian,
            include_stochastic,
            include_active,
            ham_tmpl,
            preferred_prep_basis_signs,
            preferred_meas_basis_signs,
        )

    fidpairs = [
        (x.to_circuit(prepDict), y.to_circuit(measDict)) for x, y in pauli_fidpairs
    ]  # e.g. convert ("XY","ZX") to tuple of Circuits

    listOfExperiments = []
    for (
        prepFid,
        measFid,
    ) in (
        fidpairs
    ):  # list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
        for L in max_lengths:
            listOfExperiments.append(prepFid + GiStr * L + measFid)

    return listOfExperiments


def make_idle_tomography_lists(
    nqubits,
    max_lengths,
    pauli_basis_dicts,
    maxweight=2,
    idle_string=((),),
    include_hamiltonian=True,
    include_stochastic=True,
    include_active=True,
    ham_tmpl="auto",
    preferred_prep_basis_signs="auto",
    preferred_meas_basis_signs="auto",
):
    """
    Construct lists of experiments, one for each maximum-length value, needed
    to perform idle tomography.  This is potentiall useful for running GST on
    idle tomography data.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    max_lengths : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    include_hamiltonian, include_stochastic, include_active : bool, optional
        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
        and active-type errors.

    ham_tmpl : tuple, optional
        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
        "Y", or "Z"), describing how to construct the fiducial pairs used to
        detect Hamiltonian errors.  The special (and default) value "auto"
        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
        `maxweight` equal to 1 and 2, repectively, and will generate an error
        if `maxweight > 2`.

    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
        A 3-tuple of "+" or "-" strings indicating which sign for preparing
        or measuring in the X, Y, and Z bases is preferable.  Usually one
        orientation if preferred because it's easier to achieve using the
        native model.  Additionally, the special (and default) value "auto"
        may be used, in which case :function:`preferred_signs_from_paulidict`
        is used to choose preferred signs based on `pauli_basis_dicts`.

    Returns
    -------
    list
        A list of lists of :class:`Circuit` objects, one list per max-L value.
    """

    prepDict, measDict = pauli_basis_dicts
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    GiStr = _Circuit(idle_string, num_lines=nqubits)

    pauli_fidpairs = idle_tomography_fidpairs(
        nqubits,
        maxweight,
        include_hamiltonian,
        include_stochastic,
        include_active,
        ham_tmpl,
        preferred_prep_basis_signs,
        preferred_meas_basis_signs,
    )

    fidpairs = [
        (x.to_circuit(prepDict), y.to_circuit(measDict)) for x, y in pauli_fidpairs
    ]  # e.g. convert ("XY","ZX") to tuple of Circuits

    listOfListsOfExperiments = []
    for L in max_lengths:
        expsForThisL = []
        for (
            prepFid,
            measFid,
        ) in (
            fidpairs
        ):  # list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
            expsForThisL.append(prepFid + GiStr * L + measFid)
        listOfListsOfExperiments.append(expsForThisL)

    return listOfListsOfExperiments


# -----------------------------------------------------------------------------
# Running idle tomography
# -----------------------------------------------------------------------------


def compute_observed_err_rate(
    dataset,
    pauli_fidpair,
    pauli_basis_dicts,
    idle_string,
    observable,
    max_lengths,
    fit_order=1,
):
    """
    Extract the observed error rate from a series of experiments which prepares
    and measures in *different* Pauli basis and tracks the expectation value of
    a particular `observable`.

    Parameters
    ----------
    dataset : DataSet
        The set of data counts (observations) to use.

    pauli_fidpair : tuple
        A `(prep,measure)` 2-tuple of :class:`NQPauliState` objects specifying
        the prepation state and measurement basis.

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    idle_string : Circuit
        The Circuit representing the idle operation being characterized.

    observable : NQPauliOp
        The observable whose expectation value is being tracked.

    max_lengths : list
        A list of maximum germ-power lengths.  The seriese of sequences
        considered is `prepFiducial + idle_string^L + measFiducial`, where
        `L` ranges over the values in `max_lengths`.

    fit_order : int, optional
        The polynomial order used to fit the observed data probabilities.

    Returns
    -------
    dict
        A dictionary of information about the fit, including the observed
        error rate and the data points that were fit.
    """
    # fit expectation value of `observable` (trace over all I elements of it) to a line
    pauli_prep, pauli_meas = pauli_fidpair

    prepDict, measDict = pauli_basis_dicts
    prepFid = pauli_prep.to_circuit(prepDict)
    measFid = pauli_meas.to_circuit(measDict)

    # observable is always equal to pauli_meas (up to signs) with all but 1 or 2
    # (maxErrWt in general) of it's elements replaced with 'I', essentially just
    # telling us which 1 or 2 qubits to take the <Z> or <ZZ> expectation value of
    # (since the meas fiducial gets us in the right basis) -- i.e. the qubits to *not* trace over.
    obs_indices = [i for i, letter in enumerate(observable.rep) if letter != "I"]
    minus_sign = _np.prod([pauli_meas.signs[i] for i in obs_indices])

    def unsigned_exptn_and_weight(circuit, observed_indices):
        # compute expectation value of observable
        drow = dataset[circuit]  # dataset row
        total = drow.total

        # <Z> = 0 count - 1 count (if measFid sign is +1, otherwise reversed via minus_sign)
        if len(observed_indices) == 1:
            i = observed_indices[0]  # the qubit we care about
            cnt0 = cnt1 = 0
            for outcome, cnt in drow.counts.items():
                if outcome[0][i] == "0":
                    cnt0 += cnt  # [0] b/c outcomes are actually 1-tuples
                else:
                    cnt1 += cnt
            # total = _np.abs(cnt0) + _np.abs(cnt1)
            exptn = float(cnt0 - cnt1) / total
            # I think fp might stand for 'frequency plus' and is supposed to
            # be the frequency of the plus outcome.
            fp = 0.5 + 0.5 * float(cnt0 - cnt1 + 1) / (total + 2)

        # <ZZ> = 00 count - 01 count - 10 count + 11 count (* minus_sign)
        elif len(observed_indices) == 2:
            i, j = observed_indices  # the qubits we care about
            cnt_even = cnt_odd = 0
            for outcome, cnt in drow.counts.items():
                if outcome[0][i] == outcome[0][j]:
                    cnt_even += cnt
                else:
                    cnt_odd += cnt
            exptn = float(cnt_even - cnt_odd) / total
            fp = 0.5 + 0.5 * float(cnt_even - cnt_odd + 1) / (total + 2)

        elif len(observed_indices) == 3:
            cnt_even = cnt_odd = 0
            for outcome, cnt in drow.counts.items():
                if sum(int(outcome[0][i]) for i in range(len(outcome[0]))) % 2 == 0:
                    cnt_even += cnt
                else:
                    cnt_odd += cnt
            exptn = float(cnt_even - cnt_odd) / total
            fp = 0.5 + 0.5 * float(cnt_even - cnt_odd + 1) / (total + 2)

        else:
            raise NotImplementedError(
                "Expectation values of weight > 2 observables are not implemented!"
            )

        f = 0.5 + 0.5 * exptn

        # The issue here is that the frequency can be outside the range [0,1].
        # If it is then the weight calculation fails, in this case just do OLS.
        if not (fp > 1 or fp < 0):  # standard binomial/multinomial case
            wt = _np.sqrt(total) / _np.sqrt(fp * (1.0 - fp))
            err = 2 * _np.sqrt(
                f * (1.0 - f) / total
            )  # factor of 2 b/c expectation is addition of 2 terms
        else:  # set the weight and error to None.
            wt = None
            err = None

        return exptn, wt, err

    # Get data to fit and weights to use in fitting
    data_to_fit = []
    wts = []
    errbars = []
    for L in max_lengths:
        opstr = prepFid + idle_string * L + measFid
        exptn, wt, err = unsigned_exptn_and_weight(opstr, obs_indices)
        data_to_fit.append(minus_sign * exptn)
        wts.append(wt)
        errbars.append(err)

    # curvefit -> slope
    # check that the weights are not all None (if they are this is non-CP and use OLS)
    if all([elem is not None for elem in wts]):
        coeffs = _np.polyfit(
            max_lengths, data_to_fit, fit_order, w=wts
        )  # when fit_order = 1 = line
    else:
        coeffs = _np.polyfit(
            max_lengths, data_to_fit, fit_order
        )  # when fit_order = 1 = line
    if fit_order == 1:
        slope = coeffs[0]
    elif fit_order == 2:
        # OLD: slope =  coeffs[1] # c2*x2 + c1*x + c0 ->deriv@x=0-> c1
        det = coeffs[1] ** 2 - 4 * coeffs[2] * coeffs[0]
        slope = -_np.sign(coeffs[0]) * _np.sqrt(det) if det >= 0 else coeffs[1]
        # c2*x2 + c1*x + c0 ->deriv@y=0-> 2*c2*x0 + c1;
        # x0=[-c1 +/- sqrt(c1^2 - 4c2*c0)] / 2*c2; take smaller root
        # but if determinant is < 0, fall back to x=0 slope
    else:
        raise NotImplementedError("Only fit_order <= 2 are supported!")

    return {
        "rate": slope,
        "fit_order": fit_order,
        "fitCoeffs": coeffs,
        "data": data_to_fit,
        "errbars": errbars,
        "weights": wts,
    }


def do_idle_tomography(
    nqubits,
    dataset,
    max_lengths,
    pauli_basis_dicts,
    maxweight=2,
    idle_string=((),),
    advanced_options=None,
    verbosity=0,
):
    """
    Analyze `dataset` using the idle tomography protocol to characterize
    `idle_string`.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    dataset : DataSet
        The set of data counts (observations) to use.

    max_lengths : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    advanced_options : dict, optional
        A dictionary of optional advanced arguments which influence the
        way idle tomography is performed.  Allowed keys are:

        - "preferred_prep_basis_signs" : 3-tuple of "+"/"-" or default="auto"
        - "preferred_meas_basis_signs" : 3-tuple of "+"/"-" or default="auto"
        - "pauli_fidpairs": alternate list of pauli fiducial pairs to use
        - "fit order" : integer order for polynomial fits to data
        - "ham_tmpl" : see :function:`make_idle_tomography_list`
        - "include_hamiltonian", "include_stochastic", "include_correlation",
          "include_active" : {True, False}, (default True).
           Whether to extract Hamiltonian, Stochastic, Correlation, and Active-type
           intrinsic errors.

    verbosity : int, optional
        How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    IdleTomographyResults
    """

    printer = _VerbosityPrinter.create_printer(verbosity)

    if advanced_options is None:
        advanced_options = {}

    if "pauli_fidpairs" in advanced_options:
        all_fidpairs = dict(enumerate(advanced_options["pauli_fidpairs"]))
    # print(f"{all_fidpairs = }")
    # quit()

    if nqubits == 1:  # special case where line-labels may be ('*',)
        if len(dataset) > 0:
            first_circuit = list(dataset.keys())[0]
            line_labels = first_circuit.line_labels
        else:
            line_labels = (0,)
        GiStr = _Circuit(idle_string, line_labels=line_labels)
    else:
        GiStr = _Circuit(idle_string, num_lines=nqubits)
    ## TODO: Jacobian building shouldn't generate terms for preps in things like IZ or measure in things like XI
    ## Can I just pass experiment design fiducial pairs to the jacobian building? (this is the better way)
    hamiltonian_jacobian_coefs = build_class_jacobian("H", nqubits)
    # print(hamiltonian_jacobian_coefs)

    print("Hamiltonian Jacobian coefs found")
    # print(f"{hamiltonian_jacobian_coefs = }")
    # for key in hamiltonian_jacobian_coefs.keys():
    #     print(key[1])
    # quit()

    hamiltonian_jacobian, hamiltonian_index_list = dict_to_jacobian(
        hamiltonian_jacobian_coefs, "H", nqubits, all_fidpairs
    )
    # print(f"{hamiltonian_jacobian = }")
    # print(f"{hamiltonian_index_list = }")
    # print(hamiltonian_jacobian[:,4])
    # quit()

    print("Hamiltonian Jacobian built")

    # print(hamiltonian_jacobian_coefs)
    # print(hamiltonian_index_list)
    # print(hamiltonian_jacobian)
    # print(hamiltonian_jacobian.shape)

    stochastic_jacobian_coefs = build_class_jacobian("S", nqubits)

    print("Stochastic Jacobian coefs found")

    stochastic_jacobian, stochastic_index_list = dict_to_jacobian(
        stochastic_jacobian_coefs, "S", nqubits, all_fidpairs
    )
    print("Stochastic Jacobian built")
    # quit()
    # print(stochastic_jacobian_coefs)
    correlation_jacobian_coefs = build_class_jacobian("C", nqubits)
    print("C Jacobian coefs found")
    correlation_jacobian, correlation_index_list = dict_to_jacobian(
        correlation_jacobian_coefs, "C", nqubits, all_fidpairs
    )
    print("C jacobian built")
    # print(f"{correlation_index_list = }")
    # print(f"{correlation_jacobian_coefs = }")
    # quit()
    # print(len(correlation_index_list))
    # quit()
    # print(correlation_jacobian_coefs)
    anti_symmetric_jacobian_coefs = build_class_jacobian("A", nqubits)
    print("A Jacobian coefs found")
    anti_symmetric_jacobian, anti_symmetric_index_list = dict_to_jacobian(
        anti_symmetric_jacobian_coefs, "A", nqubits, all_fidpairs
    )
    print("A Jacobian built")

    # print(hamiltonian_jacobian_coefs)
    # print(anti_symmetric_jacobian_coefs)
    error_gen_index_list = {
        "hamiltonian": [],
        "stochastic": [],
        "correlation": [],
        "anti-symmetric": [],
    }
    error_gen_index_list["hamiltonian"] += (
        "".join(["H", str(key)]) for key in hamiltonian_index_list.keys()
    )
    # print(f"{hamiltonian_jacobian_coefs = }")
    # print(f"{hamiltonian_index_list = }")

    # quit()

    error_gen_index_list["stochastic"] += (
        "".join(["S", str(key)]) for key in stochastic_index_list.keys()
    )
    error_gen_index_list["correlation"] += (
        "".join(["C", str(key[0]), ",", str(key[1])])
        for key in correlation_index_list.keys()
    )
    # print(error_gen_index_list)
    # quit()
    error_gen_index_list["anti-symmetric"] += (
        "".join(["A", str(key[0]), ",", str(key[1])])
        for key in anti_symmetric_index_list.keys()
    )

    full_jacobian = _np.hstack(
        (
            hamiltonian_jacobian,
            stochastic_jacobian,
            correlation_jacobian,
            anti_symmetric_jacobian,
        ),
    )
    full_jacobian_inv = _np.linalg.pinv(full_jacobian)
    print("Jacobian inverse calculated")

    hamil_jacobian_inv = _np.linalg.pinv(hamiltonian_jacobian)

    # if "pauli_fidpairs" in advanced_options:
    #     fidpair_index_dict = dict(enumerate(advanced_options["pauli_fidpairs"]))
    #     same_basis_fidpairs = dict()  # *all* qubits prep/meas in same basis
    #     diff_basis_fidpairs = dict()  # at least one doesn't
    #     for key, value in fidpair_index_dict.items():
    #         # pauli_fidpair is a (prep,meas) tuple of NQPauliState objects
    #         if value[0].rep == value[1].rep:  # don't care about sign
    #             same_basis_fidpairs[key] = value
    #         else:
    #             diff_basis_fidpairs[key] = value
    #     # print("DB: LENGTHS: same=",len(same_basis_fidpairs)," diff=",len(diff_basis_fidpairs))
    # else:
    #     same_basis_fidpairs = None  # just for
    #     diff_basis_fidpairs = None  # safety

    errors = _idttools.allerrors(nqubits, maxweight)
    fit_order = advanced_options.get("fit order", 1)
    intrinsic_rates = {}
    pauli_fidpair_dict = {}
    observed_rate_infos = {}
    observed_error_rates = {}
    obs_error_rates_by_exp = {}

    # pull the include_hamiltonian etc. values from advanced_options, if present.
    include_hamiltonian = advanced_options.get("include_hamiltonian", True)
    include_stochastic = advanced_options.get("include_stochastic", True)
    include_correlation = advanced_options.get("include_correlation", True)
    include_active = advanced_options.get("include_active", True)

    obs_infos = dict()

    for ifp, pauli_fidpair in all_fidpairs.items():
        # print(f"{ifp = }")
        # print(f"{pauli_fidpair = }")
        all_observables = _idttools.all_full_length_observables(
            pauli_fidpair[1], nqubits
        )
        # print(f"{all_observables = }")
        infos_for_this_fidpair = _collections.OrderedDict()
        for j, out in enumerate(all_observables):
            printer.log("  - observable %d of %d" % (j, len(all_observables)), 2)
            # print(f"{j = }")
            # print(f"{out = }")
            info = compute_observed_err_rate(
                dataset,
                pauli_fidpair,
                pauli_basis_dicts,
                GiStr,
                out,
                max_lengths,
                fit_order,
            )

            info["jacobian row"] = full_jacobian[ifp]
            infos_for_this_fidpair[out] = info
            obs_infos[ifp] = infos_for_this_fidpair
            observed_error_rates[ifp] = [
                info["rate"] for info in infos_for_this_fidpair.values()
            ]
            obs_error_rates_by_exp[pauli_fidpair] = [
                info["rate"] for info in infos_for_this_fidpair.values()
            ]
    # print(f"{obs_error_rates_by_exp = }")
    # print(f"{obs_infos = }")
    # print(f"{observed_error_rates = }")

    # if include_stochastic:
    #     if "pauli_fidpairs" in advanced_options:
    #         pauli_fidpairs = same_basis_fidpairs
    #     else:
    #         pauli_fidpairs = idle_tomography_fidpairs(
    #             nqubits,
    #             maxweight,
    #             False,
    #             include_stochastic,
    #             include_active,
    #             advanced_options.get("ham_tmpl", "auto"),
    #             preferred_prep_basis_signs,
    #             preferred_meas_basis_signs,
    #         )
    #     # print("DB: %d same-basis pairs" % len(pauli_fidpairs))

    #     obs_infos = dict()
    #     ## FIXME: through 1565 -- computing observed error rates per qubit.  i.e., extant code assumes we are checking for HX
    #     ## on qubit 0 and qubit 1 (and if weight-2 errors are allowed, then HXX as well).  However, we are now passing full-length
    #     ## pauli strings (i.e., HIX to measure the effects of HX on qubit 1), so we do not need the additional iterations
    #     for i, (ifp, pauli_fidpair) in enumerate(same_basis_fidpairs.items()):
    #         # NOTE: pauli_fidpair is a 2-tuple of NQPauliState objects

    #         all_observables = _idttools.allobservables(pauli_fidpair[1], maxweight)
    #         # all_observables = _idttools.alloutcomes(
    #         #    pauli_fidpair[0], pauli_fidpair[1], maxweight
    #         # )
    #         # print("all_observables: \n", all_observables)
    #         infos_for_this_fidpair = _collections.OrderedDict()
    #         for j, out in enumerate(all_observables):
    #             printer.log("  - observable %d of %d" % (j, len(all_observables)), 2)
    #             print(f"{j = }")
    #             print(f"{out = }")
    #             info = compute_observed_err_rate(
    #                 dataset,
    #                 pauli_fidpair,
    #                 pauli_basis_dicts,
    #                 GiStr,
    #                 out,
    #                 max_lengths,
    #                 fit_order,
    #             )
    #             info["jacobian row"] = full_jacobian[i]
    #             infos_for_this_fidpair[out] = info

    #         obs_infos[ifp] = infos_for_this_fidpair
    #         # print("infos for this fidpair:\n", infos_for_this_fidpair)
    #         # if we need additional bookkeeping, more dictionaries
    #         observed_error_rates[ifp] = [
    #             info["rate"] for info in infos_for_this_fidpair.values()
    #         ]
    #         # print("observed_error_rates:\n", observed_error_rates)
    #         printer.log(
    #             "Stochastic fidpair %d of %d: %d outcomes analyzed"
    #             % (i, len(same_basis_fidpairs.values()), len(all_observables)),
    #             1,
    #         )

    #     if include_active:
    #         pauli_fidpair_dict["samebasis"] = pauli_fidpairs  # "key" to observed rates
    #         observed_rate_infos["samebasis"] = obs_infos

    #     printer.log("Completed Stochastic/active.", 1)

    # if include_active and not include_stochastic:
    #     raise ValueError(
    #         "Cannot extract active error rates without also extracting stochastic ones!"
    #     )

    # if include_hamiltonian:
    #     if "pauli_fidpairs" in advanced_options:
    #         pauli_fidpairs = diff_basis_fidpairs
    #     else:
    #         pauli_fidpairs = idle_tomography_fidpairs(
    #             nqubits,
    #             maxweight,
    #             include_hamiltonian,
    #             False,
    #             False,
    #             advanced_options.get("ham_tmpl", "auto"),
    #             preferred_prep_basis_signs,
    #             preferred_meas_basis_signs,
    #         )
    #     # print("DB: %d diff-basis pairs" % len(pauli_fidpairs))

    #     obs_infos = dict()
    #     for i, (ifp, pauli_fidpair) in enumerate(diff_basis_fidpairs.items()):
    #         all_observables = _idttools.allobservables(pauli_fidpair[1], maxweight)

    #         infos_for_this_fidpair = _collections.OrderedDict()
    #         for j, obs in enumerate(all_observables):
    #             printer.log("  - observable %d of %d" % (j, len(all_observables)), 2)

    #             info = compute_observed_err_rate(
    #                 dataset,
    #                 pauli_fidpair,
    #                 pauli_basis_dicts,
    #                 GiStr,
    #                 obs,
    #                 max_lengths,
    #                 fit_order,
    #             )
    #             info["jacobian row"] = full_jacobian[i]
    #             infos_for_this_fidpair[obs] = info

    #         obs_infos[ifp] = infos_for_this_fidpair
    #         observed_error_rates[ifp] = [
    #             info["rate"] for info in infos_for_this_fidpair.values()
    #         ]
    #         printer.log(
    #             "Hamiltonian fidpair %d of %d: %d observables analyzed"
    #             % (i, len(diff_basis_fidpairs.values()), len(all_observables)),
    #             1,
    #         )

    #     pauli_fidpair_dict["diffbasis"] = pauli_fidpairs  # give "key" to observed rates
    #     observed_rate_infos["diffbasis"] = obs_infos
    #     printer.log("Completed Hamiltonian.", 1)

    obs_err_rates = _np.concatenate(
        [
            _np.array(
                [
                    observed_error_rates[i]
                    for i in range(len(advanced_options["pauli_fidpairs"]))
                ]
            )
        ]
    )
    # 311
    # print(obs_err_rates[311])
    # print(hamiltonian_jacobian[311,0:15])
    # quit()

    testing_intrinsic_rates = _np.dot(hamil_jacobian_inv, obs_err_rates)
    print(f"{testing_intrinsic_rates = }")
    # hs_jac = _np.hstack([hamiltonian_jacobian, stochastic_jacobian])
    # hs_jac_inv = _np.linalg.pinv(hs_jac)
    # hs_intrinsic_rates = _np.dot(hs_jac_inv, obs_err_rates)
    # print(f"{hs_intrinsic_rates = }")
    # quit()

    # print(error_gen_index_list)
    # print(sum([len(val) for val in error_gen_index_list.values()]))

    ##FIXME -- I think this only works for one qubit because of err[0] so we will come back to this
    intrinsic_rate_list = _np.dot(full_jacobian_inv, obs_err_rates)
    # print(f"{intrinsic_rate_list = }")

    # print(f'{len(intrinsic_rate_list)}')
    # quit()
    # intrinsic_rates = {k: [v for v in error_gen_index_list[k]] for k in error_gen_index_list.keys()}
    # print(error_gen_index_list)
    intrinsic_rates = {}
    # for k in error_gen_index_list.keys():
    # for v in error_gen_index_list[k]:
    # print(f'{k=}')
    # print(f'{v=}')

    intrinsic_rates = {
        v: 0 for k in error_gen_index_list.keys() for v in error_gen_index_list[k]
    }
    for key, i in zip(intrinsic_rates.keys(), range(intrinsic_rate_list.shape[0])):
        if i < 30:
            with _np.printoptions(precision=10):
                print(f"{key}:  {intrinsic_rate_list[i]}")
        intrinsic_rates[key] = intrinsic_rate_list[i]
    # with _np.printoptions(precision=10):
    # print(intrinsic_rates)

    return _IdleTomographyResults(
        dataset,
        max_lengths,
        maxweight,
        fit_order,
        pauli_basis_dicts,
        GiStr,
        errors,
        intrinsic_rates,
        pauli_fidpair_dict,
        observed_error_rates,
    )
