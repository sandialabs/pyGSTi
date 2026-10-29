"""
Dense-matrix numerical reference implementations for error generator propagation.

The functions in this module construct dense representations of error generators and
their compositions. They are primarily intended for validation and testing infrastructure.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations
import warnings as _warnings
try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator propagation tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    _warnings.warn(msg)

from math import factorial
from typing import Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pygsti.errorgenpropagation.errorpropagator as _epropagator

import numpy as _np
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis, ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL, ElementaryErrorgenLabel as _EEL
from pygsti.circuits import Circuit as _Circuit
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_str as _bel_str, canonicalize_errorgen_layer as _canonicalize_errorgen_layer
from pygsti.tools.optools import create_elementary_errorgen_nqudit, state_to_dmvec
from .corrections import stabilizer_pauli_expectation
from .sensitivities import random_support, _bitstring_to_int

__all__ = [
    'errorgen_layer_to_matrix',
    'error_generator_commutator_numerical',
    'error_generator_composition_numerical',
    'error_generator_anticommutator_numerical',
    'pauli_conjugation_composition_numerical',
    'bch_numerical',
    'pairwise_bch_numerical',
    'magnus_numerical',
    'errorgen_pauli_action_numerical',
    'zassenhaus_formula_numerical',
    'iterative_error_generator_composition_numerical',
    'phi_numerical',
    'alpha_numerical',
    'alpha_pauli_numerical',
    'stabilizer_pauli_expectation_correction_numerical',
    'approximate_stabilizer_pauli_expectation_numerical',
    'error_generator_taylor_expansion_numerical',
]


def errorgen_layer_to_matrix(errorgen_layer: Union[list[tuple[_EEL, float]], tuple[tuple[_EEL, float], ...], dict[_EEL, float]],
                             num_qubits: int, errorgen_matrix_dict: Optional[dict[_EEL, _np.ndarray]] = None,
                             sslbls: Optional[Union[list, tuple]] = None) -> _np.ndarray:
    """
    Converts an iterable over error generator coefficients and rates into the corresponding
    dense numpy array representation.

    Parameters
    ----------
    errorgen_layer : list, tuple or dict
        An iterable over error generator coefficient and rates. If a list or a tuple the
        elements should correspond to two-element tuples, the first value being an `ElementaryErrorgenLabel`
        and the second value the rate. If a dictionary the keys should be `ElementaryErrorgenLabel` and the
        values the rates.

    num_qubits : int
        Number of qubits for the error generator matrix being constructed.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.
    
    sslbls : list or tuple, optional (default None)
        A tuple or list of state space labels corresponding to the qubits upon which the error generators
        can supported. Only required when passing in a value of `errorgen_matrix_dict` with
        `GlobalElementaryErrogenLabel` keys in conjunction with an `errorgen_layer` with labels
        which are `LocalElementaryErrorgenLabel` (or vice-versa).
    
    Returns
    -------
    errorgen_mat : ndarray 
        ndarray for the dense representation of the specified error generator in the standard basis.
    """

    # if the list is empty return all zeros
    # initialize empty array for accumulation.
    mat = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    if not errorgen_layer:
        return mat
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    # infer the correct label type.
    if errorgen_matrix_dict:
        first_label = next(iter(errorgen_matrix_dict))
        if isinstance(first_label, _LEEL):
            label_type = 'local'
        elif isinstance(first_label, _GEEL):
            label_type = 'global'
        else:
            msg = f'Label type {type(first_label)} is not supported as a key for errorgen_matrix_dict.'\
                  + 'Please use either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel.'
            raise ValueError()
    else:
        raise ValueError('Non-empty errorgen_layer, but errorgen_matrix_dict is empty. Cannot convert.')
        
    # loop through errorgen_layer and accumulate the weighted error generators prescribed.
    if isinstance(errorgen_layer, (list, tuple)):
        first_coefficient_lbl = errorgen_layer[0][0]
        errorgen_layer_iter = errorgen_layer
    elif isinstance(errorgen_layer, dict):
        first_coefficient_lbl = next(iter(errorgen_layer))
        errorgen_layer_iter = errorgen_layer.items()
    else:
        raise ValueError(f'errorgen_layer should be either a list, tuple or dict. {type(errorgen_layer)=}')

    if ((isinstance(first_coefficient_lbl, _LEEL) and label_type == 'global') \
        or (isinstance(first_coefficient_lbl, _GEEL) and label_type == 'local')) and sslbls is None:
        msg = "You have passed in an `errogen_layer` with `LocalElementaryErrorgenLabel` coefficients, and " \
              +"an `errorgen_matrix_dict` with keys which are `GlobalElementaryErrorgenLabel` (or vice-versa). When using this "\
              +"combination you must also specify the state space labels with `sslbls`."
        raise ValueError(msg)

    if isinstance(first_coefficient_lbl, _LSE):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_local_eel()]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_global_eel()]
    elif isinstance(first_coefficient_lbl, _LEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_GEEL.cast(lbl, sslbls=sslbls)]
    elif isinstance(first_coefficient_lbl, _GEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_LEEL.cast(lbl, sslbls=sslbls)]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
    else:
        raise ValueError('The coefficient labels in `errorgen_layer` should be either `LocalStimErrorgenLabel`, `LocalElementaryErrorgenLabel` or `GlobalElementaryErrorgenLabel`.')
    
    return mat


# Helper functions for doing numeric commutators, compositions and BCH.

def error_generator_commutator_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL, _LSE],
                                         errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                         num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the commutator of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `ElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the commutator of the input error generators in the standard basis.
    """

    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]\
                  - errorgen_matrix_dict[errorgen2.to_local_eel()]@errorgen_matrix_dict[errorgen1.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]\
                  - errorgen_matrix_dict[_LSE.cast(errorgen2)]@errorgen_matrix_dict[_LSE.cast(errorgen1)]
    return comm

def error_generator_composition_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL, _LSE],
                                          errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                          num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the composition of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the composition of the input error generators in the standard basis.
        
    """
    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]
    return comp


def error_generator_anticommutator_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL,_LSE],
                                             errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                             num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the anticommutator of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator anticommutator being computed. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the anticommutator of the input error generators in the standard basis.
    """
    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."

    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')

        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))

    if isinstance(first_label, _LEEL):
        key1, key2 = (errorgen1, errorgen2) if isinstance(errorgen1, _LEEL) else (errorgen1.to_local_eel(), errorgen2.to_local_eel())
    else:
        key1, key2 = (errorgen1, errorgen2) if isinstance(errorgen1, _LSE) else (_LSE.cast(errorgen1), _LSE.cast(errorgen2))
    mat1, mat2 = errorgen_matrix_dict[key1], errorgen_matrix_dict[key2]
    return mat1@mat2 + mat2@mat1


def pauli_conjugation_composition_numerical(pauli: Union[str, stim.PauliString], errorgen: Union[_LEEL, _LSE],
                                            errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                            num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the composition of the Pauli conjugation superoperator rho -> Q rho Q
    with the specified elementary error generator, i.e. the dense representation of
    Q[errorgen[.]]. The conjugation is formed as the dense stochastic generator plus the
    identity, Q = S_Q + 1.

    Parameters
    ----------
    pauli : str or stim.PauliString
        The conjugating Pauli Q, as an 'I'-padded string such as 'XIZ' or a `stim.PauliString`
        (its sign is ignored). Must not be the identity.

    errorgen : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        The error generator being conjugated.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of Q[errorgen[.]] in the standard basis.
    """
    assert isinstance(errorgen, (_LEEL, _LSE))
    pauli_str = _bel_str(pauli) if isinstance(pauli, stim.PauliString) else pauli
    assert set(pauli_str) != {'I'}, "The conjugating Pauli must not be the identity."

    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')

        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))

    if isinstance(first_label, _LEEL):
        s_key = _LEEL('S', (pauli_str,))
        x_key = errorgen if isinstance(errorgen, _LEEL) else errorgen.to_local_eel()
    else:
        s_key = _LSE('S', (stim.PauliString(pauli_str),))
        x_key = errorgen if isinstance(errorgen, _LSE) else _LSE.cast(errorgen)
    s_mat = errorgen_matrix_dict[s_key]
    conjugation = s_mat + _np.eye(s_mat.shape[0])
    return conjugation@errorgen_matrix_dict[x_key]


def bch_numerical(propagated_errorgen_layers: list[_np.ndarray],
                  error_propagator: _epropagator.ErrorGeneratorPropagator,
                  bch_order: int = 1) -> _np.ndarray:
    """
    Iteratively compute effective error generator layer produced by applying the BCH approximation
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of numpy.ndarrays
        List of the error generator layers to combine using the BCH approximation (in circuit ordering)

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the BCH calculation.

    bch_order : int, optional (default 1)
        Order of the BCH approximation to apply (up to 5 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of the iterative application of the BCH
        approximation.
    """
    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    errorgen_lbl_matrix_dict = {lbl:mat for lbl,mat in zip(errorgen_basis.labels, errorgen_basis.elemgen_matrices)}
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    # initialize a matrix for storing the result of doing BCH.
    bch_result = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    if len(errorgen_layer_mats)==1:
        return errorgen_layer_mats[0]
        
    # otherwise iterate through in reverse order (the propagated layers are
    # in circuit ordering and not matrix multiplication ordering at the moment)
    # and combine the terms pairwise
    combined_err_layer = errorgen_layer_mats[-1]
    for i in range(len(errorgen_layer_mats)-2, -1, -1):
        combined_err_layer = pairwise_bch_numerical(combined_err_layer, errorgen_layer_mats[i], order=bch_order)
        
    return combined_err_layer  

def pairwise_bch_numerical(mat1: _np.ndarray, mat2: _np.ndarray, order: Literal[1,2,3,4,5]=1) -> _np.ndarray[_np.complex128]:
    """
    Helper function for doing the numerical BCH in a pairwise fashion. Note this function is primarily intended
    for numerical validations as part of testing infrastructure.
    """
    if not 1 <= order <= 5:
        raise ValueError('BCH order must be between 1 and 5 (inclusive), got %s' % str(order))

    bch_result = _np.zeros(mat1.shape, dtype=_np.complex128)
    if order >= 1:
        bch_result += mat1 + mat2
    if order >= 2:
        commutator12 = _matrix_commutator(mat1, mat2)
        bch_result += 0.5*commutator12
    if order >= 3:
        commutator112 = _matrix_commutator(mat1, commutator12)
        commutator212 = _matrix_commutator(mat2, commutator12)
        bch_result += (1/12)*(commutator112-commutator212)
    if order >= 4:
        commutator2112 = _matrix_commutator(mat2, commutator112)
        bch_result += (-1/24)*commutator2112
    if order == 5:
        commutator1112 = _matrix_commutator(mat1, commutator112)
        commutator2212 = _matrix_commutator(mat2, commutator212)
        
        commutator22212 = _matrix_commutator(mat2, commutator2212)
        commutator11112 = _matrix_commutator(mat1, commutator1112)
        commutator12212 = _matrix_commutator(mat1, commutator2212)
        commutator21112 = _matrix_commutator(mat2, commutator1112)
        commutator21212 = _matrix_commutator(mat2, _matrix_commutator(mat1, commutator212))
        commutator12112 = _matrix_commutator(mat1, commutator2112)
        
        bch_result += (-1/720)*(commutator11112 - commutator22212)
        bch_result += (1/360)*(commutator21112 - commutator12212)
        bch_result += (1/120)*(commutator21212 - commutator12112)
    return bch_result

def magnus_numerical(propagated_errorgen_layers: list[dict[_EEL, float]], error_propagator: _epropagator.ErrorGeneratorPropagator, 
                     magnus_order: Literal[1,2,3] = 1) -> _np.ndarray:
    """
    Compute effective error generator layer produced by applying the magnus expansions
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of dictionaries
        List of the error generator layers (in circuit ordering) in the form of dictionaries
        whose keys are elementary error generator labels and whose values are their corresponding
        rates. These dictionaries are in the format produced by the `ErrorGeneratorPropagator` class's
        `propagate_errorgens` method.

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the Magnus calculation.

    magnus_order : int, optional (default 1)
        Order of the Magnus expansion to apply (up to 3 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of Magnus expansion.
    """

    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    # initialize a matrix for storing the result of doing magnus.
    magnus = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    for curr_order in range(magnus_order):
        # first-order magnus terms:
        # \sum_{t1} A_{t1}
        if curr_order == 0:
            for mat in errorgen_layer_mats:
                magnus += mat
        
        # second-order magnus terms:
        # (1/2) \sum_{t1=1}^n \sum_{t2=1}^{t2} [A(t1), A(t2)]
        elif curr_order == 1:
            errorgen_pairs = []
            for i in range(len(errorgen_layer_mats)):
                for j in range(i):
                    errorgen_pairs.append((errorgen_layer_mats[i], errorgen_layer_mats[j]))
            for errorgen_pair in errorgen_pairs:
                magnus += .5*_matrix_commutator(errorgen_pair[0], errorgen_pair[1])
        
        # third-order magnus terms:
        # (1/6) \sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} ([A(t1), [A(t2),A(t3)]] + [A(t3), [A(t2), A(t1)]])
        elif curr_order == 2:
            for i in range(len(errorgen_layer_mats)):
                for j in range(i+1):
                    for k in range(j+1):
                        if i==j:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        if j==k:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
        else:
            raise NotImplementedError('Magnus beyond third order is not currently implemented.')
        
    return magnus  

def errorgen_pauli_action_numerical(errorgen: _EEL, pauli: stim.PauliString) -> _np.ndarray:
    """
    Apply the specified error generator to a given Pauli operator. This implementation
    performs the application of the error generator numerically and is primarily
    intended for use in testing. 

    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel` or numpy.ndarray
        A label specifying the error generator which should be applied to the specified Pauli operator
        or else a dense numpy array for the error generator (in the standard basis).
    
    pauli : stim.PauliString
        The pauli operator to apply the error generator to.
    
    
    Returns:
    --------
    numpy.ndarray
        Dense representation of the weighted pauli operator resulting from the application
        of the specified error generator to the input pauli.
    """

    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)

    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    pauli_unitary_shape = pauli_unitary.shape

    weighted_pauli = (errorgen_superop@pauli_unitary.ravel()).reshape(pauli_unitary_shape)

    return weighted_pauli

def zassenhaus_formula_numerical(errorgen_groups: list[dict[_EEL, float]], error_propagator: _epropagator.ErrorGeneratorPropagator, 
                                 zassenhaus_order: Literal[1,2] = 1) -> list[_np.ndarray]:
    """
    Function for numerically computing the nth-order Zassenhaus formula for a set of error generators.
    Please see https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula#Zassenhaus_formula
    for more information on this approximation.

    Due to the numerical nature of this implementation it is not meant for efficient computation and
    primarily supports testing.

    Parameters
    ----------
    errorgen_groups : list of dicts
        List of dictionaries of the error generator coefficients and rates for a group of error generators corresponding
        to each of the operators in the sum to perform Zassenhaus with respect to. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    zassenhaus_order : int, optional (default 1)
        Order of the Zassenahaus formula to compute, currently supports up to second order.
    
    Returns
    -------
    zassenhaus_formula_arrays : list of numpy.ndarrays
        A list of numpy arrays, each corresponding to one of the operators which is exponentiated in 
        the product as output by the Zassenhaus formula.    
    """

    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in errorgen_groups:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_group_mats = []
    for layer in errorgen_groups:
        errorgen_group_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))

    zassenhaus_formula_arrays = []

    if zassenhaus_order>=1:
        if zassenhaus_order==1:
            return errorgen_group_mats
        else:
            zassenhaus_formula_arrays.extend(errorgen_group_mats)

    if zassenhaus_order>=2:
        second_order_correction_array = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
        errorgen_pairs = []
        for i in range(len(errorgen_groups)):
            for j in range(i):
                errorgen_pairs.append((errorgen_group_mats[i], errorgen_group_mats[j]))
        for errorgen_pair in errorgen_pairs:
            second_order_correction_array += .5*_matrix_commutator(errorgen_pair[0], errorgen_pair[1])
        zassenhaus_formula_arrays.append(second_order_correction_array)

    if zassenhaus_order>=3:
        raise NotImplementedError('The Zassenhaus formula is currently only implemented up to second-order.')    

    return zassenhaus_formula_arrays
    
def _matrix_commutator(mat1: _np.ndarray, mat2: _np.ndarray) -> _np.ndarray:
    return mat1@mat2 - mat2@mat1

def iterative_error_generator_composition_numerical(errorgen_labels: tuple[_LEEL, ...], rates: tuple[float, ...],
                                                    errorgen_matrix_dict: Optional[dict[_LEEL, _np.ndarray]] = None,
                                                    num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Iteratively compute error generator compositions. The function computes a dense representation of this composition
    numerically and is primarily intended as part of testing infrastructure.
    
    Parameters
    ----------
    errorgen_labels : tuple of `LocalElementaryErrorgenLabel`s
        A tuple of the elementary error generator labels to be composed.
    
    rates : tuple of float
        A tuple of corresponding error generator rates of the same length as the tuple
        of error generator labels.
        
    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    numpy.ndarray
        Dense numpy array representation of the super operator corresponding to the iterated composition written in 
        the standard basis.
    """
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    composition = errorgen_matrix_dict[errorgen_labels[0]]
    for lbl in errorgen_labels[1:]:
        composition = composition@errorgen_matrix_dict[lbl]
    composition *= _np.prod(rates)
    return composition


# helper function for numerically computing phi, primarily used for testing.
def phi_numerical(tableau: stim.Tableau, desired_bitstring: str, P: Union[str, stim.PauliString],
                  Q: Union[str, stim.PauliString]) -> _np.ndarray:
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    (This version does this calculation numerically and is primarily intended for testing infrastructure.)
    
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
    
    # start by getting the pauli string which maps the all-zeros string to the target bitstring.
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring])).to_unitary_matrix(endian = 'big')

    # map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    stabilizer_state = tableau.to_state_vector(endian = 'big')
    stabilizer_state.reshape((len(stabilizer_state),1))
    # combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string@P.to_unitary_matrix(endian = 'big')
    eff_Q = Q.to_unitary_matrix(endian = 'big')@initial_pauli_string
    
    # now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    # the corresponding phase corrections.
    # all_zeros = '0'*len(eff_P)
    all_zeros = _np.zeros((2**len(desired_bitstring),1))
    all_zeros[0] = 1  
    # calculate phi.
    # The second amplitude also needs a complex conjugate applied
    phi = (all_zeros.T@eff_P@stabilizer_state) * (stabilizer_state.conj().T@eff_Q@all_zeros)
    
    num_random = random_support(tableau)
    scale = 2**(num_random)

    return phi*scale

def alpha_numerical(errorgen: Union[_LSE, _EEL], tableau: stim.Tableau, desired_bitstring: str) -> float:
    """
    First-order error generator sensitivity function for probability. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    # get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    # also need a superbra for the desired bitstring.
    desired_bitstring_vec = _np.zeros(2**len(desired_bitstring))
    desired_bitstring_vec[_bitstring_to_int(desired_bitstring)] = 1
    desired_bitstring_dmvec = state_to_dmvec(desired_bitstring_vec)
    desired_bitstring_dmvec.reshape((1, len(desired_bitstring_dmvec)))
    num_random = random_support(tableau)
    scale = 2**(num_random)
    
    # compute the needed trace inner product.
    alpha = _np.real_if_close(scale*(desired_bitstring_dmvec.conj().T@errorgen_superop@stabilizer_state_dmvec))
    
    return alpha

def alpha_pauli_numerical(errorgen: Union[_LSE, _LEEL], tableau: stim.Tableau, pauli: stim.PauliString):
    """
    First-order error generator sensitivity function for pauli expectations. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.
    """
    
    # get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    # finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    # flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    # compute the needed trace inner product.
    alpha = _np.real_if_close(pauli_vec.conj().T@errorgen_superop@stabilizer_state_dmvec).item()
    
    return alpha

def stabilizer_pauli_expectation_correction_numerical(errorgen_dict: dict[_EEL, float],
                                                      errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                                      circuit: _Circuit, pauli: stim.PauliString,
                                                      order: int = 1) -> float:
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        Circuit the expectation value is being measured against.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    tableau = circuit.convert_to_stim_tableau()
    
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    
    # also get the superoperator (in the standard basis) corresponding to the taylor series
    # expansion of the specified error generator dictionary.
    taylor_expanded_errorgen = error_generator_taylor_expansion_numerical(errorgen_dict, errorgen_propagator, order=order, mx_basis='std')
    
    # finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    # flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    expectation_correction = _np.linalg.multi_dot([pauli_vec.conj().T, taylor_expanded_errorgen,stabilizer_state_dmvec]).item()
    return expectation_correction

def approximate_stabilizer_pauli_expectation_numerical(errorgen_dict: dict[_EEL, float],
                                                       errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                                       circuit: _Circuit, pauli: stim.PauliString,
                                                       order: int = 1) -> float:
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.
    This function performs the corrections numerically and so it primarily intended for testing
    infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        A pygsti `Circuit` or a stim.Tableau to compute the output pauli expectation value for.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value for.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
    Returns
    -------
    expectation_value : float
        Approximate expectation value for desired pauli.
    """
    
    tableau = circuit.convert_to_stim_tableau()

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    ideal_expectation = stabilizer_pauli_expectation(tableau, pauli)
    correction = stabilizer_pauli_expectation_correction_numerical(errorgen_dict, errorgen_propagator, circuit, pauli, order)
    return ideal_expectation + correction

def error_generator_taylor_expansion_numerical(errorgen_dict: dict[_EEL, float],
                                               errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                               order: int = 1,
                                               mx_basis: Union[str, _Basis] = 'pp') -> _np.ndarray:
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity). This function computes a dense representation
    of this taylor expansion as a numpy array and is primarily intended for testing infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    mx_basis : `Basis` or str, optional (default 'pp')
        Basis in which to return the matrix.

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the nth order taylor expansion of the specified error generator.
    """
       
    errorgen_mat = errorgen_propagator.errorgen_layer_dict_to_errorgen(errorgen_dict, mx_basis)
    taylor_expansion = _np.zeros(errorgen_mat.shape, dtype=_np.complex128)
    for i in range(1, order+1):
        taylor_expansion += 1/factorial(i)*_np.linalg.matrix_power(errorgen_mat, i)

    return taylor_expansion
