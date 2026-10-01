"""
Utility functions relevant to Lindblad forms and projections
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Literal, Optional, Union, overload

Literal_HSCA = Literal['H', 'S', 'C', 'A']
Literal_HO = Literal['H', 'O']

import numpy as _np
import scipy.sparse as _sps

from pygsti.tools.basistools import basis_matrices
from pygsti.tools import sparsepsd as _sparsepsd
from pygsti.tools.graphs import sparsechol as _sparsechol
from pygsti.tools.legacytools import warn_deprecated as _warn_deprecated
from pygsti.baseobjs.basis import (
    Basis as _Basis,
    TensorProdBasis as _TensorProdBasis,
    BuiltinBasis as _BuiltinBasis
)
from pygsti.baseobjs.errorgenlabel import (
    GlobalElementaryErrorgenLabel as _GEEL,
    LocalElementaryErrorgenLabel as _LEEL,
    _TOKEN_REGEX,
    _bel_tokens
)


from pygsti.baseobjs.statespace import (
    QubitSpace as _QubitSpace,
    StateSpace as _StateSpace
)


def create_elementary_errorgen_dual(typ: Literal_HSCA, p: _np.ndarray, q: Optional[_np.ndarray]=None, 
                                    sparse: bool=False, normalization_factor: Optional[Union[float, str]]='auto') -> Union[_np.ndarray, _sps.csr_array]:
    """
    Construct a "dual" elementary error generator matrix in the "standard" (matrix-unit) basis.

    The elementary error generator that is dual to the one computed by calling
    :func:`create_elementary_errorgen` with the same argument.  This dual element
    can be used to find the coefficient of the original, or "primal" elementary generator.
    For example, if `A = sum(c_i * E_i)`, where `E_i` are the elementary error generators given
    by :func:`create_elementary_errorgen`), then `c_i = dot(D_i.conj(), A)` where `D_i`
    is the dual to `E_i`.

    There are four different types of dual elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j/(2d^2) * [ p, rho ]`
    Stochastic:   `L(rho) = 1/(d^2) p * rho * p^\\dag`
    Correlation:  `L(rho) = 1/(2d^2) ( p * rho * q^\\dag + q * rho * p^\\dag)`
    Active:       `L(rho) = 1j/(2d^2) ( p * rho * q^\\dag - q * rho * p^\\dag)`

    where `d` is the dimension of the Hilbert space, e.g. 2 for a single qubit.  Square
    brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of dual error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    normalization_factor : str or float, optional (default 'auto')
        String or float specifying the normalization factor to apply. If
        a string the options are 'auto' and 'auto_return', which both use
        the corresponding (primal) elementary error generator to calculate
        this automatically and only differ in whether they return this 
        normalization factor. If a float, the reciprocal of the input value
        is used directly.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0]; d2 = d**2
    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    #if p or q is a sparse matrix fall back to original implementation
    if not isinstance(p, _np.ndarray) or (q is not None and not isinstance(q, _np.ndarray)):
        for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
            # Only difference between H/S/C/A is how they transform input density matrices
            if typ == 'H':
                rho1 = -1j * (p @ rho0 - rho0 @ p)  # -1j / (2 * d2) *
            elif typ == 'S':
                rho1 = (p @ rho0 @ pdag)  # 1 / d2 *
            elif typ == 'C':
                rho1 = (p @ rho0 @ qdag + q @ rho0 @ pdag)  # 1 / (2 * d2) *
            elif typ == 'A':
                rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag)  # 1j / (2 * d2)
            elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
    else:
        # Loop through the standard basis as all possible input density matrices
        for i in range(d):  # rho0 == input density mx
            for j in range(d):
                # Only difference between H/S/C/A is how they transform input density matrices
                if typ == 'H':
                    rho1 = _np.zeros((d,d), dtype=_np.complex128)
                    rho1[:, j] = -1j*p[:, i]
                    rho1[i, :] += 1j*p[j, :]
                elif typ == 'S':
                    rho1 = p[:,i].reshape((d,1))@pdag[j,:].reshape((1,d))
                elif typ == 'C':
                    rho1 = p[:,i].reshape((d,1))@qdag[j,:].reshape((1,d)) + q[:,i].reshape((d,1))@pdag[j,:].reshape((1,d))
                elif typ == 'A':
                    rho1 = 1j*(p[:,i].reshape((d,1))@ qdag[j,:].reshape((1,d))) - 1j*(q[:,i].reshape((d,1))@pdag[j,:].reshape((1,d)))

                elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    return_normalization = bool(normalization_factor == 'auto_return')
    if normalization_factor in ('auto', 'auto_return'):
        primal = create_elementary_errorgen(typ, p, q, sparse)
        if sparse:
            normalization_factor = _np.vdot(elem_errgen.toarray().flatten(), primal.toarray().flatten())
        else:
            normalization_factor = _np.vdot(elem_errgen.flatten(), primal.flatten())
    elem_errgen *= _np.real_if_close(1 / normalization_factor).item()  # item() -> scalar
    if sparse: elem_errgen = elem_errgen.tocsr()
    return (elem_errgen, normalization_factor) if return_normalization else elem_errgen


#TODO: Should be able to leverage the structure of the paulis as generalized permutation
#matrices to avoid explicitly doing outer products
def create_elementary_errorgen_dual_pauli(typ: Literal_HSCA, p: _np.ndarray, q: Optional[_np.ndarray]=None, 
                                          sparse: bool=False) -> Union[_np.ndarray, _sps.csr_array]:
    """
    Construct a "dual" elementary error generator matrix in the "standard" (matrix-unit) basis.
    Specialized to p and q being elements of the (unnormalized) pauli basis.

    The elementary error generator that is dual to the one computed by calling
    :func:`create_elementary_errorgen` with the same argument.  This dual element
    can be used to find the coefficient of the original, or "primal" elementary generator.
    For example, if `A = sum(c_i * E_i)`, where `E_i` are the elementary error generators given
    by :func:`create_elementary_errorgen`), then `c_i = dot(D_i.conj(), A)` where `D_i`
    is the dual to `E_i`.

    There are four different types of dual elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j/(2d^2) * [ p, rho ]`
    Stochastic:   `L(rho) = 1/(d^2) p * rho * p`
    Correlation:  `L(rho) = 1/(2d^2) ( p * rho * q + q * rho * p)`
    Active:       `L(rho) = 1j/(2d^2) ( p * rho * q - q * rho * p)`

    where `d` is the dimension of the Hilbert space, e.g. 2 for a single qubit.  Square
    brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of dual error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0]; d2 = d**2

    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    #if p or q is a sparse matrix fall back to original implementation
    if not isinstance(p, _np.ndarray) or (q is not None and not isinstance(q, _np.ndarray)):
        for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
            # Only difference between H/S/C/A is how they transform input density matrices
            if typ == 'H':
                rho1 = -1j * (p @ rho0 - rho0 @ p)  # -1j / (2 * d2) *
            elif typ == 'S':
                rho1 = (p @ rho0 @ p)  # 1 / d2 *
            elif typ == 'C':
                rho1 = (p @ rho0 @ q + q @ rho0 @ p)  # 1 / (2 * d2) *
            elif typ == 'A':
                rho1 = 1j * (p @ rho0 @ q - q @ rho0 @ p)  # 1j / (2 * d2)
            elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
    else:
        if typ == 'H':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d): 
                for j in range(d):
                    rho1 = _np.zeros((d,d), dtype=_np.complex128)
                    rho1[:, j] = -1j*p[:, i]
                    rho1[i, :] += 1j*p[j, :]
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        elif typ == 'S':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d):
                for j in range(d):
                    rho1 = p[:,i].reshape((d,1))@p[j,:].reshape((1,d))
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        elif typ == 'C':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d): 
                for j in range(d):
                    rho1 = p[:,i].reshape((d,1))@q[j,:].reshape((1,d)) + q[:,i].reshape((d,1))@p[j,:].reshape((1,d))
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        else:
            # Loop through the standard basis as all possible input density matrices
            for i in range(d):  
                for j in range(d):
                    rho1 = 1j*(p[:,i].reshape((d,1))@ q[j,:].reshape((1,d))) - 1j*(q[:,i].reshape((d,1))@p[j,:].reshape((1,d)))
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if typ in 'HCA':
        normalization_factor = 1/(2*d2)
    else:
        normalization_factor = 1/d2

    elem_errgen *= normalization_factor
    if sparse: elem_errgen = elem_errgen.tocsr()
    return elem_errgen


#TODO: The construction can be made a bit more efficient if we know we will be constructing multiple
#error generators with overlapping indices by reusing intermediate results.
def create_elementary_errorgen(typ : Literal_HSCA, p: _np.ndarray, 
                               q: Optional[_np.ndarray]=None, sparse: bool=False) -> Union[_np.ndarray, _sps.csr_array]:    
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.

    There are four different types of elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j * [ p, rho ]`
    Stochastic:   `L(rho) = p * rho * p^\\dag - 0.5*{p^\\dag p, rho}`
    Correlation:  `L(rho) = p * rho * q^\\dag + q * rho * p^\\dag - 0.5 {(p^\\dag @ q + q^\\dag @ p), rho}`
    Active:       `L(rho) = 1j( p * rho * q^\\dag - q * rho * p^\\dag + 0.5 {(p^\\dag @ q - q^\\dag @ p)), rho} )`

    Square brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    eeg_dtype : _np.typing.DTypeLike
    if typ == 'H':
        assert q is None, "q must not be provided for H-type elementary error generator!"
        eeg_dtype = _np.complex128
    elif typ == 'S':
        assert q is None, "q must not be provided for S-type elementary error generator!"
        eeg_dtype = p.dtype
    elif typ == 'C':
        assert q is not None, "q must be provided for C-type elementary error generator!"
        eeg_dtype = _np.result_type(p.dtype, q.dtype)
    elif typ == 'A':
        assert q is not None, "q must be provided for A-type elementary error generator!"
        eeg_dtype = _np.complex128
    else:
        raise ValueError(f'`typ` must be one of "H", "S", "C", or "A"; received {typ}.')

    d = p.shape[0]
    d2 = d**2
    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=eeg_dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=eeg_dtype)

    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if typ in 'CA':
        pq_plus_qp  = pdag @ q + qdag @ p
        pq_minus_qp = pdag @ q - qdag @ p
    if typ in 'S':
        pdag_p = pdag @ p

    # if p or q is a sparse matrix fall back to original implementation
    if not isinstance(p, _np.ndarray) or (q is not None and not isinstance(q, _np.ndarray)):
        # Loop through the standard basis as all possible input density matrices
        for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
            # Only difference between H/S/C/A is how they transform input density matrices
            if typ == 'H':
                rho1 = -1j * (p @ rho0 - rho0 @ p)  # Add "/2" to have PP ham gens match previous versions of pyGSTi
            elif typ == 'S':
                rho1 = p @ rho0 @ pdag - 0.5 * (pdag_p @ rho0 + rho0 @ pdag_p)
            elif typ == 'C':
                rho1 = p @ rho0 @ qdag + q @ rho0 @ pdag - 0.5 * (pq_plus_qp @ rho0 + rho0 @ pq_plus_qp)
            elif typ == 'A':
                rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag + 0.5 * (pq_minus_qp @ rho0 + rho0 @ pq_minus_qp))
            elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
    else:
        # Loop through the standard basis as all possible input density matrices
        rho1 = _np.zeros((d,d), dtype=eeg_dtype)
        for i in range(d): 
            for j in range(d):
                # Only difference between H/S/C/A is how they transform input density matrices
                if typ == 'H':
                    # rho1 complex
                    rho1[:] = 0
                    rho1[:, j]  = -1j*p[:, i]  # plain assignment
                    rho1[i, :] +=  1j*p[j, :]  # in-place update
                elif typ == 'S':
                    # rho1 same dtype as p
                    rho1[:] = p[:,i].reshape((d,1)) @ pdag[j,:].reshape((1,d))
                    rho1[:, j] -= 0.5*pdag_p[:, i]
                    rho1[i, :] -= 0.5*pdag_p[j, :]
                elif typ == 'C':
                    # rho1 the result dtype of (p, q)
                    rho1[:] = p[:,i].reshape((d,1)) @ qdag[j,:].reshape((1,d)) + q[:,i].reshape((d,1)) @ pdag[j,:].reshape((1,d))
                    rho1[:, j] -= 0.5*pq_plus_qp[:, i]
                    rho1[i, :] -= 0.5*pq_plus_qp[j, :]
                elif typ == 'A':
                    # rho1 complex
                    rho1[:] = 1j*(p[:,i].reshape((d,1)) @ qdag[j,:].reshape((1,d))) - 1j*(q[:,i].reshape((d,1)) @ pdag[j,:].reshape((1,d)))
                    rho1[:, j] += 0.5j * pq_minus_qp[:, i]
                    rho1[i, :] += 0.5j * pq_minus_qp[j, :]

                elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse:
        elem_errgen = elem_errgen.tocsr()

    return elem_errgen


#TODO: Should be able to leverage the structure of the paulis as generalized permutation
#matrices to avoid explicitly doing outer products
def create_elementary_errorgen_pauli(typ: Literal_HSCA, p: _np.ndarray, q: Optional[_np.ndarray]=None, 
                                     sparse: bool=False, normalized_paulis: bool=False) -> Union[_np.ndarray, _sps.csr_array]:    
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.
    Specialized to the case where p and q are elements of the pauli basis.

    There are four different types of elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j * [ p, rho ]`
    Stochastic:   `L(rho) = p * rho * p - rho`
    Correlation:  `L(rho) = p * rho * q + q * rho * p - 0.5 {{p,q}, rho}`
    Active:       `L(rho) = 1j( p * rho * q - q * rho * p + 0.5 {[p,q], rho} )`

    Square brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    normalized_paulis : bool, optional
        Whether `p` (and `q`) are *normalized* Pauli-basis elements (scaled so that
        Tr(P^2) = 1) rather than the unnormalized convention (P^2 = I).  When True, the
        trace-preserving correction term of the 'S' (stochastic) generator is scaled by
        1/d so the generator stays consistent with the normalized basis.  Only affects
        'S'-type generators; the H/C/A constructions already scale correctly with `p`/`q`.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0]
    d2 = d**2
    # For normalized Paulis, B^dag B = I/d, so the S-generator's trace-preserving term scales by 1/d.
    rho_scale = (1 / d) if normalized_paulis else 1
    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    #should be able to get away with just doing one product here.
    if typ in 'CA':
        pq = p@q
        qp = q@p
        pq_plus_qp = pq + qp
        pq_minus_qp = pq - qp

    #if p or q is a sparse matrix fall back to original implementation
    if not isinstance(p, _np.ndarray) or (q is not None and not isinstance(q, _np.ndarray)):
        # Loop through the standard basis as all possible input density matrices
        for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
            # Only difference between H/S/C/A is how they transform input density matrices
            if typ == 'H':
                rho1 = -1j * (p @ rho0 - rho0 @ p)  # Add "/2" to have PP ham gens match previous versions of pyGSTi
            elif typ == 'S':
                rho1 = p @ rho0 @ p - rho_scale * rho0
            elif typ == 'C':
                rho1 = p @ rho0 @ q + q @ rho0 @ p - 0.5 * (pq_plus_qp @ rho0 + rho0 @ pq_plus_qp)
            elif typ == 'A':
                rho1 = 1j * (p @ rho0 @ q - q @ rho0 @ p + 0.5 * (pq_minus_qp @ rho0 + rho0 @ pq_minus_qp))
            elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
    else:
        if typ == 'H':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d):  
                for j in range(d):
                    rho1 = _np.zeros((d,d), dtype=_np.complex128)
                    rho1[:, j] = -1j*p[:, i]
                    rho1[i, :] += 1j*p[j, :]
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        elif typ == 'S':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d): 
                for j in range(d):
                    rho1 = p[:,i].reshape((d,1))@p[j,:].reshape((1,d))
                    rho1[i,j] -= rho_scale
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        elif typ == 'C':
            # Loop through the standard basis as all possible input density matrices
            for i in range(d): 
                for j in range(d):
                    rho1 = p[:,i].reshape((d,1))@q[j,:].reshape((1,d)) + q[:,i].reshape((d,1))@p[j,:].reshape((1,d))
                    rho1[:, j] += -0.5*pq_plus_qp[:, i]
                    rho1[i, :] += -0.5*pq_plus_qp[j, :]
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        else:
            # Loop through the standard basis as all possible input density matrices
            for i in range(d): 
                for j in range(d):
                    rho1 = 1j*(p[:,i].reshape((d,1))@ q[j,:].reshape((1,d))) - 1j*(q[:,i].reshape((d,1))@p[j,:].reshape((1,d)))
                    rho1[:, j] += 1j*.5*pq_minus_qp[:, i]
                    rho1[i, :] += 1j*.5*pq_minus_qp[j, :]
                    elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: elem_errgen = elem_errgen.tocsr()
    return elem_errgen


def create_lindbladian_term_errorgen(typ: Literal_HO, Lm: _np.ndarray, Ln: Optional[_np.ndarray]=None, 
                                     sparse: bool=False) -> Union[_np.ndarray, _sps.csr_array]:  # noqa N803
    """
    Construct the superoperator for a term in the common Lindbladian expansion of an error generator.

    Mathematically, for d-dimensional matrices Lm and Ln, this routine
    constructs the d^2-dimension Lindbladian matrix L whose action is
    given by:

    `L(rho) = -i [Lm, rho] `    (when `typ == 'H'`)

    or

    `L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)`    (`typ == 'O'`)

    where rho is a density matrix.  L is returned as a superoperator
    matrix that acts on a vectorized density matrices.

    Parameters
    ----------
    typ : {'H', 'O'}
        The type of error generator to construct.

    Lm : numpy.ndarray
        d-dimensional basis matrix.

    Ln : numpy.ndarray, optional
        d-dimensional basis matrix.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = Lm.shape[0] 
    d2 = d**2
    if sparse:
        lind_errgen = _sps.lil_matrix((d2, d2), dtype=Lm.dtype)
    else:
        lind_errgen = _np.empty((d2, d2), dtype=Lm.dtype)

    assert(typ in ('H', 'O')), "`typ` must be one of 'H' or 'O'"
    assert((typ in 'H' and Ln is None) or (typ in 'O' and Ln is not None)), \
        "Wrong number of basis elements provided for %s-type lindblad term errorgen!" % typ

    if typ in 'O':
        Lm_dag = _np.conjugate(_np.transpose(Lm))
        Lmdag_Ln = Lm_dag @ Ln

    # Loop through the standard basis as all possible input density matrices
    for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
        # Only difference between H/S/C/A is how they transform input density matrices
        if typ == 'H':
            rho1 = -1j * (Lm @ rho0 - rho0 @ Lm)
        else: # typ == 'O':
            rho1 = Ln @ rho0 @ Lm_dag - 0.5 * (Lmdag_Ln @ rho0 + rho0 @ Lmdag_Ln)
        lind_errgen[:, i] = rho1.ravel()
        # ^ That line used to branch based on the value of sparse, but both branches
        #   produced the same result.

    if sparse:
        lind_errgen = lind_errgen.tocsr()
    return lind_errgen


def _as_generator(seed) -> _np.random.Generator:
    """
    Convert `seed` to a numpy Generator without touching NumPy's global random state.

    A Generator is returned unchanged, and a BitGenerator is wrapped with its algorithm and state
    intact. None, integer seed material, and SeedSequences follow `numpy.random.default_rng`. A
    legacy RandomState shares its underlying bit generator (as `default_rng` does natively from
    NumPy 2.2 on), though not its legacy normal-distribution stream.
    """
    if isinstance(seed, _np.random.RandomState):
        seed = seed._bit_generator
    return _np.random.default_rng(seed)


def _check_basis_factor(basis) -> None:
    """
    Raise ValueError unless `basis` has an identity first element and Hermitian, traceless,
    mutually trace-orthogonal other elements.
    """
    if isinstance(basis, _TensorProdBasis):
        for factor in basis.component_bases:
            _check_basis_factor(factor)
        return
    elements = basis.elements  # Force lazy construction so invalid builtin dimensions still fail.
    if len(elements) == 0:
        raise ValueError("Basis %s has no elements." % basis.name)
    if isinstance(basis, _BuiltinBasis) and basis.name in ('pp', 'PP', 'gm', 'GM', 'gm_unnormalized'):
        return  # Their registered constructors establish the required properties.
    els = [el.toarray() if _sps.issparse(el) else _np.asarray(el) for el in elements]
    d = els[0].shape[0]
    E = _np.array([el.reshape(-1) for el in els])
    scale = _np.max(_np.abs(E))
    tol = 1e-10 * max(scale * scale * d, 1.0)
    if not _np.allclose(els[0], els[0][0, 0] * _np.eye(d), atol=1e-10 * scale) or abs(els[0][0, 0]) == 0:
        raise ValueError("The first element of basis %s must be proportional to the identity." % basis.name)
    for lbl, el in zip(basis.labels[1:], els[1:]):
        if not _np.allclose(el, el.conj().T, atol=1e-10 * scale):
            raise ValueError("Element %s of basis %s is not Hermitian." % (lbl, basis.name))
        if abs(_np.trace(el)) > tol:
            raise ValueError("Element %s of basis %s is not traceless." % (lbl, basis.name))
    G = E.conj() @ E.T
    if _np.max(_np.abs(G - _np.diag(_np.diag(G)))) > tol:
        raise ValueError("The elements of basis %s are not trace-orthogonal." % basis.name)


def _resolve_errorgen_basis(state_space: _StateSpace, elementary_errorgen_basis: _Basis | str | None) -> _Basis:
    """ The validated operator basis whose non-identity elements are the Lindblad directions. """
    # A module-level import would cycle through errorgenbasis -> optools -> lindbladtools.
    from pygsti.baseobjs.errorgenbasis import canonical_errorgen_basis, _check_errorgen_state_space

    if elementary_errorgen_basis is None:
        return canonical_errorgen_basis(state_space)
    _check_errorgen_state_space(state_space)
    if isinstance(elementary_errorgen_basis, str):
        try:
            basis = _Basis.cast(elementary_errorgen_basis, state_space)
            for factor in (basis.component_bases if isinstance(basis, _TensorProdBasis) else [basis]):
                factor.elements  # elements are built lazily, and may fail only now
        except (ValueError, AssertionError) as e:
            raise ValueError("The basis %r is not available on %s: %s"
                             % (elementary_errorgen_basis, state_space, e)) from None
    elif isinstance(elementary_errorgen_basis, _Basis):
        basis = elementary_errorgen_basis
    else:
        raise TypeError("`elementary_errorgen_basis` must be None, a basis name, or a Basis, not %s."
                        % type(elementary_errorgen_basis).__name__)
    if basis.dim != state_space.dim:
        raise ValueError("Basis %s has dimension %d, but the state space %s has dimension %d."
                         % (basis.name, basis.dim, state_space, state_space.dim))
    # A tensor product of valid factors is valid, and much cheaper to check.
    for factor in (basis.component_bases if isinstance(basis, _TensorProdBasis) else [basis]):
        _check_basis_factor(factor)
    return basis


def _direction_supports(basis: _Basis, state_space: _StateSpace) -> _np.ndarray | None:
    """
    Mark each non-identity basis element's subsystem support, or return None if unknown.

    Rows follow ``basis.labels[1:]`` and columns follow the state-space subsystem order.
    Labels must contain one PP/GM token per subsystem. A tensor-product basis must also
    have one factor per subsystem, with matching dimensions.

    Examples
    --------
    For two qubits, IX and XI each act on one subsystem, while XX acts on both:

    >>> from pygsti.baseobjs import Basis, QubitSpace
    >>> space = QubitSpace(2)
    >>> basis = Basis.cast('PP', space)
    >>> supports = _direction_supports(basis, space)
    >>> [supports[list(basis.labels).index(lbl) - 1].tolist() for lbl in ('IX', 'XI', 'XX')]
    [[False, True], [True, False], [True, True]]

    Arbitrary labels do not identify subsystem support, even if the matrices are Pauli matrices:

    >>> from pygsti.baseobjs import ExplicitBasis
    >>> basis = ExplicitBasis(Basis.cast('PP', 4).elements, labels=('identity', 'a', 'b', 'c'))
    >>> _direction_supports(basis, QubitSpace(1)) is None
    True
    """
    sslbls = state_space.sole_tensor_product_block_labels
    if isinstance(basis, _TensorProdBasis):
        if [c.dim for c in basis.component_bases] != [state_space.label_dimension(l) for l in sslbls]:
            return None
    tokens = [_bel_tokens(lbl) for lbl in basis.labels[1:]]
    if not all(len(t) == len(sslbls) and all(_TOKEN_REGEX.fullmatch(x) for x in t) for t in tokens):
        return None
    return _np.array([[x != 'I' for x in t] for t in tokens], dtype=bool).reshape(len(tokens), len(sslbls))


def _fixed_rates_by_index(fixed_errorgen_rates, sslbls, index) -> dict:
    """
    Sort fixed rates by type. Returns {'H': {a: (lbl, v)}, 'S': ..., 'C': {(a, b): (lbl, v)},
    'A': ...}, with a, b indices of non-identity basis elements and lbl the local label.
    """
    fixed = {'H': {}, 'S': {}, 'C': {}, 'A': {}}
    for key, v in (fixed_errorgen_rates or {}).items():
        if isinstance(key, _GEEL):
            lbl = _LEEL.cast(key, sslbls=sslbls)
        elif isinstance(key, _LEEL):
            lbl = key
        else:
            raise TypeError("Keys of `fixed_errorgen_rates` must be elementary error generator labels, not %s."
                            % type(key).__name__)
        missing = [bel for bel in lbl.basis_element_labels if bel not in index]
        if missing:
            raise ValueError("Fixed rate %s refers to %s, which are not non-identity labels of the error generator "
                             "basis." % (key, missing))
        idx = tuple(index[bel] for bel in lbl.basis_element_labels)
        if lbl.errorgen_type in ('C', 'A') and idx[0] == idx[1]:
            raise ValueError("Fixed rate %s pairs a basis element with itself." % str(key))
        k = idx[0] if lbl.errorgen_type in ('H', 'S') else tuple(sorted(idx))
        if k in fixed[lbl.errorgen_type]:
            raise ValueError("The %s rate for %s is fixed more than once." % (lbl.errorgen_type, str(key)))
        fixed[lbl.errorgen_type][k] = (lbl, float(v))
    return fixed


SeedLike = Union[None, int, Sequence[int], _np.random.SeedSequence, _np.random.BitGenerator,
                 _np.random.Generator, _np.random.RandomState]


@overload
def random_cptp_errorgen_rates(
        state_space: _StateSpace, *,
        errorgen_types: tuple[Literal_HSCA, ...] = ...,
        elementary_errorgen_basis: _Basis | str | None = ...,
        max_weights: Mapping[Literal_HSCA, int] | None = ...,
        sslbl_overlap: Collection[Hashable] | None = ...,
        H_params: tuple[float, float] = ...,
        SCA_params: tuple[float, float] = ...,
        error_metric: Literal['generator_infidelity', 'total_generator_error'] | None = ...,
        error_metric_value: float | None = ...,
        relative_HS_contribution: tuple[float, float] | None = ...,
        fixed_errorgen_rates: Mapping[_LEEL | _GEEL, float] | None = ...,
        label_type: Literal['global'] = ...,
        seed: SeedLike = ...,
    ) -> dict[_GEEL, float]: ...


@overload
def random_cptp_errorgen_rates(
        state_space: _StateSpace, *,
        errorgen_types: tuple[Literal_HSCA, ...] = ...,
        elementary_errorgen_basis: _Basis | str | None = ...,
        max_weights: Mapping[Literal_HSCA, int] | None = ...,
        sslbl_overlap: Collection[Hashable] | None = ...,
        H_params: tuple[float, float] = ...,
        SCA_params: tuple[float, float] = ...,
        error_metric: Literal['generator_infidelity', 'total_generator_error'] | None = ...,
        error_metric_value: float | None = ...,
        relative_HS_contribution: tuple[float, float] | None = ...,
        fixed_errorgen_rates: Mapping[_LEEL | _GEEL, float] | None = ...,
        label_type: Literal['local'],
        seed: SeedLike = ...,
    ) -> dict[_LEEL, float]: ...


def random_cptp_errorgen_rates(
        state_space: _StateSpace, *,
        errorgen_types: tuple[Literal_HSCA, ...] = ('H', 'S', 'C', 'A'),
        elementary_errorgen_basis: _Basis | str | None = None,
        max_weights: Mapping[Literal_HSCA, int] | None = None,
        sslbl_overlap: Collection[Hashable] | None = None,
        H_params: tuple[float, float] = (0., .01),
        SCA_params: tuple[float, float] = (0., .01),
        error_metric: Literal['generator_infidelity', 'total_generator_error'] | None = None,
        error_metric_value: float | None = None,
        relative_HS_contribution: tuple[float, float] | None = None,
        fixed_errorgen_rates: Mapping[_LEEL | _GEEL, float] | None = None,
        label_type: Literal['global', 'local'] = 'global',
        seed: SeedLike = None,
    ) -> dict[_GEEL, float] | dict[_LEEL, float]:
    """
    Sample error-generator rates whose generator exponentiates to a CPTP map on `state_space`.

    `elementary_errorgen_basis` supplies the operator directions used to construct elementary
    H, S, C and A generators. `errorgen_types`, `max_weights` and `sslbl_overlap` select which
    elementary generators are sampled. The selected S directions index the Kossakowski matrix
    K; selected C and A pairs allow its real and imaginary off-diagonal components, respectively.
    Fixed rates can extend this support as described in Notes.

    In the selected operator-basis order, K_ii = S_i and K_ij = C_ij - 1j * A_ij for i < j.
    The returned rates define a positive-semidefinite K, ensuring that the generator's
    exponential is completely positive and trace preserving, or the call raises an error.

    Parameters
    ----------
    state_space : StateSpace
        The state space the error generator acts on: a single tensor product block of quantum
        subsystems, each of dimension at least 2 (qubits, qudits, or a mixture). Its labels are
        the labels of the returned global error generator labels. Direct sums are not supported.

    errorgen_types : tuple of {'H', 'S', 'C', 'A'}, optional
        The sectors to sample. Including 'C' or 'A' requires 'S'.

    elementary_errorgen_basis : Basis or str, optional
        The full, identity-first operator basis whose non-identity elements define the H, S, C
        and A directions. Use None for the canonical PP/GM tensor-product basis, a string for a
        builtin basis family, or a Basis of dimension ``state_space.dim``. Non-identity elements
        must be Hermitian, traceless and mutually trace-orthogonal. Normalization, labels, and
        subsystem structure are described in Notes.

    max_weights : mapping, optional
        The maximum weight of each sector, keyed by 'H', 'S', 'C' and 'A'; a missing sector has no
        limit. The weight of an element is the number of subsystems on which it is not the
        identity, whatever their dimensions. The weight of a C or A pair is that of the union of
        the two elements' supports, and a pair is only allowed when both of its S directions are.

    sslbl_overlap : collection of state space labels, optional
        Require each sampled H or S direction to act on at least one of these subsystems.
        Each direction of a sampled C or A pair must itself be an allowed S direction, so both
        directions must meet this condition.

    H_params : tuple of float, optional
        The mean and standard deviation of the normal distribution of the H rates.

    SCA_params : tuple of float, optional
        The mean and standard deviation of the proposal distribution of the entries of L. The
        mean must be 0. The standard deviation sigma sets the scale of K: every expected S rate
        is sigma**2, so without an error budget the total S rate grows with the number of
        directions, which is D**2 - 1 without restrictions.

    error_metric : {'generator_infidelity', 'total_generator_error'}, optional
        A budget for the sampled rates, together with `error_metric_value`; give both or
        neither. 'generator_infidelity' is sum(h**2) + sum(s) and 'total_generator_error' is
        sum(|h|) + sum(s), over all returned H rates h and S rates s. These are coefficient
        budgets in the scale of the selected basis, not channel infidelities. For the canonical
        normalization, sum(h**2) = Tr(H**2) / D for the Hamiltonian H and sum(s) = Tr(K), so the
        generator infidelity keeps its leading-order meaning on qudits. The budget is met by
        rescaling the free (not fixed) rates: H rates by a common factor, and K by a diagonal
        congruence, which keeps it positive semidefinite. Without `relative_HS_contribution`, the
        free H and S contributions are scaled by the same factor. Nonzero `H_params` means are
        not supported together with a budget.

    error_metric_value : float, optional
        The target value of `error_metric`.

    relative_HS_contribution : tuple of float, optional
        The fractions of `error_metric_value` carried by the H and S sectors, summing to 1.
        Requires `error_metric` and both 'H' and 'S' in `errorgen_types`.

    fixed_errorgen_rates : mapping, optional
        Prescribed rates keyed by local or global elementary error generator labels from the
        selected basis. These override sampled rates and can extend the allowed sectors and
        support. The diagonal requirements and PSD feasibility checks are described in Notes.

    label_type : {'global', 'local'}, optional
        Whether the keys of the result are `GlobalElementaryErrorgenLabel` objects (on the
        labels of `state_space`) or `LocalElementaryErrorgenLabel` objects (whose basis element
        labels are the basis's own labels, as `LindbladErrorgen.from_elementary_errorgens`
        expects).

    seed : int, numpy.random.Generator, or other seed, optional
        Anything `numpy.random.default_rng` accepts: None (fresh entropy), integer seed material,
        a SeedSequence, a BitGenerator (kept, with its state), or a Generator (used directly and
        advanced). A legacy RandomState shares its bit generator. NumPy's global random state is
        never used.

    Returns
    -------
    dict
        Rates keyed by elementary error generator labels: the H rates, then the S rates, then the
        C and A rates on every allowed pair, in basis order.

    Notes
    -----
    The default operator basis is :func:`pygsti.baseobjs.canonical_errorgen_basis`: PP on qubits
    and GM on other subsystems, normalized so that Tr(F_a^dag F_b) = D delta_ab in Hilbert-space
    dimension D. A string selects a family on each subsystem through ``Basis.cast(name, state_space)``.
    GM works on all supported subsystem dimensions. PP gives the Pauli-product basis on qubits;
    it also permits higher power-of-two dimensions, but its labels then do not identify the
    given subsystem supports. Lowercase gm and pp are orthonormal instead of D-normalized,
    changing the scale of the rates and error metrics.

    Weight limits, ``sslbl_overlap`` and global labels require subsystem supports. These are
    inferred from PP/GM labels with one token per subsystem: I, X, Y, Z, X_{j,k}, Y_{j,k}, or Z_{j}.
    A tensor-product basis must have one factor of matching dimension per subsystem. For a
    basis without this structure, use ``label_type='local'`` and no support restrictions.

    When constructing a ``LindbladErrorgen`` from these rates, pass the same operator basis.
    In particular, ``LindbladErrorgen.from_elementary_errorgens(..., elementary_errorgen_basis='GM')``
    on a qubit-qutrit space uses a single 6-dimensional Gell-Mann basis with different labels.
    Pass ``canonical_errorgen_basis(state_space)`` to use this sampler's default tensor-product basis.

    K is sampled through a Cholesky factor L on the filled pattern of the allowed C/A pairs
    after a fill-reducing ordering (see :mod:`pygsti.tools.graphs.sparsechol`), with a Bartlett-type draw.
    For the unrestricted pattern, K is a scaled Wishart matrix before fixed rates or budgets
    are imposed. A chordal pattern admits an ordering without fill, so the factor draw respects
    its support. Otherwise fill entries are zeroed. Disallowed real or imaginary components
    are also zeroed, and the remaining off-diagonal entries are scaled toward zero if needed
    to restore positive semidefiniteness while retaining the sampled diagonal.

    Fixed H and S rates are included even when sector, weight or overlap restrictions would
    exclude them. A fixed C or A rate also adds both of its operator directions to K. If S is
    enabled, their diagonal rates are sampled even when support restrictions would exclude
    those directions; otherwise both S rates must be fixed explicitly. This does not enable
    other C or A pairs involving these added directions.

    A diagonal congruence imposes the fixed S values and any budget scaling of the free S
    values. Free H rates are scaled separately. The prescribed H and S rates remain unchanged.

    When fixed C or A components are present, they are assigned after the diagonals are set.
    Let B contain those diagonals and fixed components, with all other off-diagonal components zero. B must be PSD
    to numerical tolerance; an indefinite B raises ValueError even when the free components
    could complete it to a PSD matrix. This routine does not search for such a completion.

    The remaining free off-diagonal components form O, and the final matrix is B + t*O.
    For numerically positive-definite B, t is the largest feasible value in [0, 1], with a
    small safety margin when t < 1. For numerically singular B, t is zero conservatively,
    discarding the free off-diagonal components while retaining all prescribed values.
    """
    errorgen_types = tuple(errorgen_types)
    if not set(errorgen_types) <= set('HSCA'):
        raise ValueError("`errorgen_types` may only contain 'H', 'S', 'C' and 'A', not %s." % str(errorgen_types))
    if label_type not in ('global', 'local'):
        raise ValueError("Unsupported label type %r." % label_type)
    if ('C' in errorgen_types or 'A' in errorgen_types) and 'S' not in errorgen_types:
        raise ValueError("'C' and 'A' rates require 'S' rates for a positive-semidefinite Kossakowski matrix.")
    if (error_metric is None) != (error_metric_value is None):
        raise ValueError("Give both `error_metric` and `error_metric_value`, or neither.")
    if error_metric not in (None, 'generator_infidelity', 'total_generator_error'):
        raise ValueError("Unsupported error metric %r." % error_metric)
    if relative_HS_contribution is not None:
        if error_metric is None:
            raise ValueError("`relative_HS_contribution` requires `error_metric`.")
        if 'H' not in errorgen_types or 'S' not in errorgen_types:
            raise ValueError("`relative_HS_contribution` requires both 'H' and 'S' in `errorgen_types`.")
        if len(relative_HS_contribution) != 2 or min(relative_HS_contribution) < 0 \
                or abs(1 - sum(relative_HS_contribution)) > 1e-7:
            raise ValueError("`relative_HS_contribution` must be two nonnegative fractions summing to 1.")
    if SCA_params[0] != 0:
        raise ValueError("Nonzero means in `SCA_params` are not supported.")
    if error_metric is not None and H_params[0] != 0:
        raise ValueError("A nonzero mean in `H_params` is not supported together with an error metric.")
    max_weights = dict(max_weights or {})
    if not set(max_weights) <= set('HSCA'):
        raise ValueError("The keys of `max_weights` must be among 'H', 'S', 'C' and 'A'.")

    basis = _resolve_errorgen_basis(state_space, elementary_errorgen_basis)
    sslbls = state_space.sole_tensor_product_block_labels
    bels = list(basis.labels[1:])
    index = {bel: a for a, bel in enumerate(bels)}
    supports = _direction_supports(basis, state_space)
    if supports is None and (max_weights or sslbl_overlap is not None or label_type == 'global'):
        raise ValueError("Weight limits, `sslbl_overlap` and global labels need a basis with one 'PP'- or "
                         "'GM'-style factor per subsystem, which basis %s is not. Use label_type='local'." % basis.name)

    # Which directions (and, for C/A, pairs) each sector allows.
    num_dirs = len(bels)
    if supports is None:
        supports = _np.zeros((num_dirs, len(sslbls)), dtype=bool)  # unused: no restrictions apply
    weights = supports.sum(axis=1)
    if sslbl_overlap is None:
        overlaps = _np.ones(num_dirs, dtype=bool)
    else:
        unknown = [l for l in sslbl_overlap if l not in sslbls]
        if unknown:
            raise ValueError("`sslbl_overlap` contains %s, which are not labels of %s." % (unknown, state_space))
        overlaps = supports[:, [sslbls.index(l) for l in sslbl_overlap]].any(axis=1)

    def allowed_dirs(typ):
        if typ not in errorgen_types:
            return _np.zeros(num_dirs, dtype=bool)
        return overlaps & (weights <= max_weights.get(typ, _np.inf))

    fixed = _fixed_rates_by_index(fixed_errorgen_rates, sslbls, index)
    H_dirs = sorted(set(_np.flatnonzero(allowed_dirs('H'))) | set(fixed['H']))
    S_allowed = allowed_dirs('S')
    K_dirs = sorted(set(_np.flatnonzero(S_allowed)) | set(fixed['S'])
                    | {a for pairs in (fixed['C'], fixed['A']) for pair in pairs for a in pair})
    if 'S' not in errorgen_types and any(a not in fixed['S'] for a in K_dirs):
        raise ValueError("Fixed C or A rates need S rates on both of their directions: include 'S' in "
                         "`errorgen_types`, or fix those S rates.")
    K_dirs = _np.array(K_dirs, dtype=int)
    pos = {a: p for p, a in enumerate(K_dirs)}
    n = len(K_dirs)

    # Allowed C/A pairs among the allowed S directions, by the union of their supports.
    Ksupp = supports[K_dirs]
    union_weight = weights[K_dirs][:, None] + weights[K_dirs][None, :] - Ksupp.astype(int) @ Ksupp.T.astype(int)
    pair_overlap = overlaps[K_dirs][:, None] | overlaps[K_dirs][None, :]
    both_S = S_allowed[K_dirs][:, None] & S_allowed[K_dirs][None, :]
    offdiag_mask = both_S & pair_overlap & ~_np.eye(n, dtype=bool)
    real_mask = offdiag_mask & (union_weight <= max_weights.get('C', _np.inf)) & ('C' in errorgen_types)
    imag_mask = offdiag_mask & (union_weight <= max_weights.get('A', _np.inf)) & ('A' in errorgen_types)
    for mask, typ in ((real_mask, 'C'), (imag_mask, 'A')):
        for (a, b) in fixed[typ]:
            mask[pos[a], pos[b]] = mask[pos[b], pos[a]] = True

    # Sample H, then the Kossakowski matrix K: its diagonal contains S rates, and
    # K[i, j] = C[i, j] - 1j*A[i, j] for i < j. A PSD K gives a Lindblad generator
    # whose exponential is completely positive and trace preserving.
    rng = _as_generator(seed)
    h = dict(zip(H_dirs, rng.normal(loc=H_params[0], scale=H_params[1], size=len(H_dirs))))
    has_C = 'C' in errorgen_types or bool(fixed['C'])
    has_A = 'A' in errorgen_types or bool(fixed['A'])
    offdiag = 'complex' if has_C and has_A else ('imag' if has_A else 'real')
    if n > 0:
        st = _sparsechol.CholeskyStructure(real_mask | imag_mask)
        K = _sparsepsd._sample_psd(st, rng, offdiag, scale=SCA_params[1],
                                 real_pattern=real_mask, imag_pattern=imag_mask)
    else:
        K = _np.zeros((0, 0))

    # Fix H and S rates and meet the budget.
    for a, (_, v) in fixed['H'].items():
        h[a] = v
    H_free = [a for a in H_dirs if a not in fixed['H']]
    S_free = _np.array([p for p, a in enumerate(K_dirs) if a not in fixed['S']], dtype=int)
    H_scale, S_scale = 1.0, 1.0
    if error_metric is not None:
        def _free_scale(needed: float, free: float, what: str) -> float:
            """ The factor by which a free contribution `free` must scale to become `needed`. """
            tol = 1e-12 * max(abs(needed), 1.0)
            if needed < -tol:
                raise ValueError("The fixed rates alone exceed the %s target by %g." % (what, -needed))
            if needed <= tol:
                return 0.0
            if free <= 0:
                raise ValueError("No free rates can carry the remaining %s target of %g." % (what, needed))
            return needed / free

        power = 2 if error_metric == 'generator_infidelity' else 1
        fixed_H = sum(abs(v) ** power for _, v in fixed['H'].values())
        fixed_S = sum(v for _, v in fixed['S'].values())
        free_H = sum(abs(h[a]) ** power for a in H_free)
        free_S = float(_np.sum(_np.real(_np.diag(K))[S_free]))
        if relative_HS_contribution is not None:
            H_scale = _free_scale(relative_HS_contribution[0] * error_metric_value - fixed_H, free_H, 'H')
            S_scale = _free_scale(relative_HS_contribution[1] * error_metric_value - fixed_S, free_S, 'S')
        else:
            H_scale = S_scale = _free_scale(error_metric_value - fixed_H - fixed_S, free_H + free_S, error_metric)
        H_scale = H_scale ** (1 / power)
    for a in H_free:
        h[a] *= H_scale
    fixed_S_pos = [pos[a] for a in fixed['S']]
    K = _sparsepsd._impose_diagonal(K, list(S_free) + fixed_S_pos,
                         list(S_scale * _np.real(_np.diag(K))[S_free]) + [v for _, v in fixed['S'].values()])

    # Fix C and A rates; K_ij = C_ij - 1j * A_ij for i < j.
    if fixed['C'] or fixed['A']:
        K = _sparsepsd._impose_offdiagonals(K, real={(pos[a], pos[b]): v for (a, b), (_, v) in fixed['C'].items()},
                                 imag={(pos[a], pos[b]): -v if index[l.basis_element_labels[0]] == a else v
                                       for (a, b), (l, v) in fixed['A'].items()})

    # Assemble the result: fixed rates under the caller's labels (and orientation), others in basis order.
    rates = {}
    for a in H_dirs:
        rates[fixed['H'][a][0] if a in fixed['H'] else _LEEL('H', (bels[a],))] = float(h[a])
    for p, a in enumerate(K_dirs):
        rates[fixed['S'][a][0] if a in fixed['S'] else _LEEL('S', (bels[a],))] = float(_np.real(K[p, p]))
    for typ, mask, part in (('C', real_mask, _np.real), ('A', imag_mask, lambda z: -_np.imag(z))):
        for p, q in zip(*_np.nonzero(_np.triu(mask, k=1))):
            a, b = K_dirs[p], K_dirs[q]
            if (a, b) in fixed[typ]:
                lbl, v = fixed[typ][(a, b)]
                rates[lbl] = v
            else:
                rates[_LEEL(typ, (bels[a], bels[b]))] = float(part(K[p, q]))

    if label_type == 'global':
        rates = {_GEEL.cast(lbl, sslbls=sslbls): v for lbl, v in rates.items()}
    return rates


def random_CPTP_error_generator_rates(
        num_qubits        : int,
        errorgen_types    : tuple[Literal_HSCA, ...] = ('H', 'S', 'C', 'A'),
        max_weights       : dict[str, int] | None = None,
        H_params          : tuple[float, float] = (0., .01),
        SCA_params        : tuple[float, float] = (0., .01),
        error_metric      : Literal['generator_infidelity', 'total_generator_error'] | None = None,
        error_metric_value        : float | None = None,
        relative_HS_contribution  : tuple[float, float] | None = None,
        fixed_errorgen_rates      : dict[_LEEL | _GEEL, float] | None = None,
        sslbl_overlap             : list | None = None,
        label_type                : Literal['global', 'local'] = 'global',
        seed           : SeedLike = None,
        qubit_labels   : list | None = None
    ) -> dict:
    """
    Function for generating a random set of CPTP error generator rates.

    Deprecated: use :func:`random_cptp_errorgen_rates`, which takes a `StateSpace` (e.g.
    `QubitSpace(num_qubits)`) in place of `num_qubits` and `qubit_labels`, and supports qudits.
    This function now calls it with the 'PP' basis. Its sampler is new: the returned rates
    define a generator whose exponential is CPTP, and outputs for a given seed differ from
    earlier versions of pyGSTi.

    Parameters
    ----------
    num_qubits : int
        Number of qubits the error generator acts upon.

    errorgen_types : tuple of str, optional (default('H', 'S', 'C', 'A'))
        Tuple of strings designating elementary error generator types to include in this
        basis. Note that due to the CP constraint, certain values are not allowed,
        and any tuple containing 'C' or 'A' terms must also include 'S'.

    max_weights : dict, optional (default None)
        An optional dictionary specifying the maximum weight
        for each of the elementary error generator types, with keys
        given by the strings 'H', 'S', 'C' and 'A'. If None then
        there is no maximum weight. If specified, any error generator
        types without entries will have no maximum weight associated
        with them.

    H_params : tuple of floats, optional (default (0.,.01))
        Mean and standard deviation parameters for a normal distribution
        from which the H rates will be sampled. Note that specifying a non-zero
        value for the mean with generator_infidelity set to a non-trivial value
        is not supported, and will raise an error.

    SCA_params : tuple of floats, optional (default (0.,.01))
        Mean and standard deviation parameters for a normal distribution
        from which the entries of the matrix used in the construction of the S, C and A rates
        will be construction is sampled. Note that specifying a non-zero
        value for the mean with generator_infidelity set to a non-trivial value
        is not supported, and will raise an error.

    error_metric : str, optional (default None)
        An optional string, used in conjunction with the error_metric_value
        kwarg which specifies which metric to use in setting the sampled
        channel's overall error rate. If None, no target value for the channel's
        overall error rate is used. Currently supported options include:

        - 'generator_infidelity'
        - 'total_generator_error'

    error_metric_value : float, optional (default None)
        An float between 0 and 1 which gives the target value of the
        error metric specified in 'error_metric' for the channel induced by
        the randomly produced error generator. If None
        then no target value is used and the returned error generator
        will have a random generator infidelity.

    relative_HS_contribution : tuple, optional (default None)
        An optional tuple, used in conjunction with the `generator_infidelity` kwarg,
        specifying the relative contributions of the H and S error generators to the
        generator infidelity. The values in this tuple should sum to 1. The first entry
        corresponds to the H sector, and the second the S sector.

    sslbl_overlap : list of sslbls, optional (default None)
        A list of state space labels corresponding to qudits the support of
        an error generator must overlap with (i.e. the support must include at least
        one of these qudits) in order to be included in this basis.

    fixed_errorgen_rates : dict, optional (default None)
        An optional dictionary whose keys are `LocalElementaryErrorgenLabel`
        objects, and whose values are error generator rates. When specified, the
        rates in this dictionary will override any randomly selected values in the
        final returned error generator rate dictionary. The inclusion of these
        rates is performed independently of any of the kwargs which otherwise
        control the weight and allowed types of the error generators in this
        model. If the fixed C and A rates are incompatible with complete
        positivity, a ValueError is raised.

    label_type : str, optional (default 'global')
        String which can be either 'global' or 'local', indicating whether to
        return a dictionary with keys which are `GlobalElementaryErrorgenLabel`
        or `LocalElementaryErrorgenLabel` objects respectively.

    seed : int or numpy.random.Generator, optional (default None)
        An optional seed, or anything else `random_cptp_errorgen_rates` accepts.

    qubit_labels : list or int or str, optional (default None)
        An optional list of qubit labels upon which the error generator should act.
        Only utilized when returning global labels.

    Returns
    -------
    Dictionary of error generator coefficient labels and rates
    """
    _warn_deprecated('random_CPTP_error_generator_rates', 'random_cptp_errorgen_rates')
    if error_metric is None:  # these used to be ignored without a metric
        error_metric_value = relative_HS_contribution = None
    rates = random_cptp_errorgen_rates(
        _QubitSpace(num_qubits), errorgen_types=errorgen_types, elementary_errorgen_basis='PP',
        max_weights=max_weights, sslbl_overlap=sslbl_overlap, H_params=H_params, SCA_params=SCA_params,
        error_metric=error_metric, error_metric_value=error_metric_value,
        relative_HS_contribution=relative_HS_contribution, fixed_errorgen_rates=fixed_errorgen_rates,
        label_type=label_type, seed=seed)
    if label_type == 'global' and qubit_labels is not None:
        mapper = {i: lbl for i, lbl in enumerate(qubit_labels)}
        rates = {lbl.map_state_space_labels(mapper): v for lbl, v in rates.items()}
    return rates
