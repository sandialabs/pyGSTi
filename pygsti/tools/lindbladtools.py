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
from typing import Literal

Literal_HSCA = Literal['H', 'S', 'C', 'A']

import numpy as _np
import scipy.sparse as _sps

from pygsti.tools.basistools import basis_matrices
import pygsti.baseobjs as _bo
from pygsti.baseobjs.errorgenlabel import (
    GlobalElementaryErrorgenLabel as _GEEL,
    LocalElementaryErrorgenLabel as _LEEL
)


from pygsti.baseobjs.statespace import (
    QubitSpace as _QubitSpace,
    StateSpace as _StateSpace
)


def create_elementary_errorgen_dual(typ, p, q=None, sparse=False, normalization_factor='auto'):
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
def create_elementary_errorgen_dual_pauli(typ, p, q=None, sparse=False):
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
def create_elementary_errorgen(typ : Literal_HSCA, p, q=None, sparse=False):
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.

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
                    rho1[:, j] = -1j*p[:, i]
                    rho1[i, :] =  1j*p[j, :]
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
def create_elementary_errorgen_pauli(typ, p, q=None, sparse=False):
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.
    Specialized to the case where p and q are elements of the (unnormalized) pauli basis.

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

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0] 
    d2 = d**2
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
                rho1 = p @ rho0 @ p - rho0
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
                    rho1[i,j] += -1
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


def create_lindbladian_term_errorgen(typ, Lm, Ln=None, sparse=False):  # noqa N803
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
    d = Lm.shape[0]; d2 = d**2
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
        elif typ == 'O':
            rho1 = Ln @ rho0 @ Lm_dag - 0.5 * (Lmdag_Ln @ rho0 + rho0 @ Lmdag_Ln)
        else: raise ValueError("Invalid lindblad term errogen type!")
        lind_errgen[:, i] = rho1.ravel()
        # ^ That line used to branch based on the value of sparse, but both branches
        #   produced the same result.

    if sparse:
        lind_errgen = lind_errgen.tocsr()
    return lind_errgen


def _filtered_leeldict_to_array(
        d: dict[_LEEL, float],
        t: Literal_HSCA,
    ) -> _np.ndarray:
    """Return the values of d whose key's errorgen type matches t."""
    return _np.array([val for key, val in d.items() if key.errorgen_type == t])


def _validate_random_CPTP_params(
        errorgen_types       : tuple[Literal_HSCA, ...],
        H_params             : tuple[float, float],
        SCA_params           : tuple[float, float],
        error_metric         : Literal['generator_infidelity', 'total_generator_error'] | None,
        error_metric_value   : float | None,
        rel_HS_contrib       : tuple[float, float] | None,
        fixed_errorgen_rates : dict[_LEEL, float],
        max_weights          : dict[str, int] | None
    ) -> tuple[float, float]:
    """Validate inputs for random_CPTP_error_generator_rates. Returns (fixed_H_contrib, fixed_S_contrib)."""

    if 'C' in errorgen_types or 'A' in errorgen_types:
        msg = 'Must include S terms when C or A present. Cannot have a CP error generator otherwise.'
        assert 'S' in errorgen_types, msg

    if max_weights is not None:
        msg = 'The maximum weight of the %s terms should be <= the maximum weight of S.'
        assert max_weights.get('C', 0) <= max_weights.get('S', 0), msg % 'C'
        assert max_weights.get('A', 0) <= max_weights.get('S', 0), msg % 'A'

    if len(fixed_errorgen_rates) == 0 or error_metric is None:
        return 0.0, 0.0
    
    msg = 'Specifying non-zero HSCA means together with a target error metric is not supported.'
    assert H_params[0] == 0. and SCA_params[0] == 0., msg

    msg = 'error_metric_value must be specified when error_metric is not None.'
    assert error_metric_value is not None, msg
    
    msg = 'All keys of fixed_errorgen_rates must be LocalElementaryErrorgenLabel.'
    assert all([isinstance(key, _LEEL) for key in fixed_errorgen_rates.keys()]), msg
    
    fixed_S_rates = _filtered_leeldict_to_array(fixed_errorgen_rates, 'S')
    fixed_S_contrib = _np.sum(fixed_S_rates)

    fixed_H_rates = _filtered_leeldict_to_array(fixed_errorgen_rates, 'H')
    if error_metric == 'generator_infidelity':
        fixed_H_rates = fixed_H_rates ** 2
    if error_metric == 'total_generator_error':
        fixed_H_rates = _np.abs(fixed_H_rates)
    fixed_H_contrib = _np.sum(fixed_H_rates)

    fixed_error_metric_value = fixed_S_contrib + fixed_H_contrib
    msg  = f'Incompatible values of error_metric_value and fixed_errorgen_rates. The value of '
    msg += f'{error_metric}={error_metric_value} is less than the value of {fixed_error_metric_value} '
    msg += f'corresponding to the given fixed_errorgen_rates_dict.'
    assert fixed_error_metric_value <= error_metric_value, msg
    
    if rel_HS_contrib is not None:

        msg = f'Fixed %s contribution to {error_metric} of %f exceeds overall %s contribution target value of %f.'
        abs_H_contrib = rel_HS_contrib[0]*error_metric_value
        abs_S_contrib = rel_HS_contrib[1]*error_metric_value
        assert fixed_H_contrib <= abs_H_contrib, msg % ('H', fixed_H_contrib, 'H', abs_H_contrib )
        assert fixed_S_contrib <= abs_S_contrib, msg % ('S', fixed_S_contrib, 'S', abs_S_contrib )

        msg = 'Invalid rel_HS_contrib, %s is not in errorgen_types.'
        assert 'H' in errorgen_types, msg % 'H'
        assert 'S' in errorgen_types, msg % 'S'
        assert abs(1-sum(rel_HS_contrib)) <= 1e-7, 'The rel_HS_contrib should sum to 1.'

    return fixed_H_contrib, fixed_S_contrib


def _filter_ca_labels_for_cp(
        labels:   list[_LEEL],
        s_labels: list[_LEEL]
    ) -> list[_LEEL]:
    """Return only those labels whose both basis element labels have matching S labels in s_labels."""
    filtered = []
    for lbl in labels:
        comp1, comp2 = lbl.basis_element_labels
        lbl1 = _LEEL('S', (comp1,))
        lbl2 = _LEEL('S', (comp2,))
        if lbl1 in s_labels and lbl2 in s_labels:
            filtered.append(lbl)
    return filtered


def _check_ca_cp_constraint(
        ca_rates_dict: dict[_LEEL, float],
        s_rates_dict : dict[_LEEL, float],
        errgen_type: str
    ) -> None:
    """Raise ValueError if any rate in ca_rates_dict violates |rate| <= sqrt(S1 * S2)."""
    for lbl, rate in ca_rates_dict.items():
        comp1, comp2 = lbl.basis_element_labels
        lbl1 = _LEEL('S', (comp1,))
        lbl2 = _LEEL('S', (comp2,))
        rate1 = s_rates_dict[lbl1]
        rate2 = s_rates_dict[lbl2]
        if not (abs(rate) <= _np.sqrt(rate1 * rate2)):
            print(f'{lbl}: {rate}')
            raise ValueError(f'Invalid {errgen_type} rate')
    return


def _generate_random_rates(
        errgen_labels_H: list[_LEEL],
        errgen_labels_S: list[_LEEL],
        errgen_labels_C: list[_LEEL],
        errgen_labels_A: list[_LEEL],
        H_params    : tuple[float, float],
        SCA_params  : tuple[float, float],
        rng         : _np.random.Generator
    ) -> dict[Literal_HSCA, dict[_LEEL, float]]:
    """Generate random H/S/C/A rates; S/C/A come from PSD matrices to satisfy CP constraints."""
    num_H_rates = len(errgen_labels_H)
    num_S_rates = len(errgen_labels_S)
    rates = {}
    rates['H'] = {lbl: val for lbl, val in zip(errgen_labels_H,
                  rng.normal(loc=H_params[0], scale=H_params[1], size=num_H_rates))}
    random_SC_gen_mat = rng.normal(loc=SCA_params[0], scale=SCA_params[1], size=(num_S_rates, num_S_rates))
    random_SA_gen_mat = rng.normal(loc=SCA_params[0], scale=SCA_params[1], size=(num_S_rates, num_S_rates))
    random_SC_mat = random_SC_gen_mat @ random_SC_gen_mat.T
    random_SA_mat = random_SA_gen_mat @ random_SA_gen_mat.T
    random_S_rates = _np.real(_np.diag(random_SC_mat) + _np.diag(random_SA_mat))
    rates['S'] = {lbl: val for lbl, val in zip(errgen_labels_S, random_S_rates)}
    rates['C'] = {lbl: val for lbl, val in zip(errgen_labels_C,
                  random_SC_mat[_np.triu_indices_from(random_SC_mat, k=1)])}
    rates['A'] = {lbl: val for lbl, val in zip(errgen_labels_A,
                  random_SA_mat[_np.triu_indices_from(random_SA_mat, k=1)])}
    return rates


def _rescale_to_error_metric(
        random_rates_dicts: dict[Literal_HSCA, dict[_LEEL, float]],
        error_metric: Literal['generator_infidelity', 'total_generator_error'],
        error_metric_value: float,
        relative_HS_contribution: tuple[float, float] | None,
        fixed_H_contrib: float,
        fixed_S_contrib: float,
        errorgen_types: tuple[str, ...],
        H_free_keys: list[_LEEL],
        S_free_keys: list[_LEEL],
        C_free_keys: list[_LEEL],
        A_free_keys: list[_LEEL]
    ) -> None:
    """Scale free H/S/C/A rates so the overall error metric equals error_metric_value (mutates random_rates_dicts)."""

    current_S_sum_free = 0.0 
    current_H_sum_free = 0.0

    if 'S' in errorgen_types:
        rrd = random_rates_dicts['S']
        current_S_sum_free = _np.sum( [ rrd[key] for key in S_free_keys ] )

    if 'H' in errorgen_types:
        rrd = random_rates_dicts['H']
        if error_metric == 'generator_infidelity':
            current_H_sum_free = _np.sum( [ rrd[key] ** 2 for key in H_free_keys ] )
        if error_metric == 'total_generator_error':
            current_H_sum_free = _np.sum( [ abs(rrd[key]) for key in H_free_keys ] )

    total_H_sum = current_H_sum_free + fixed_H_contrib
    total_S_sum = current_S_sum_free + fixed_S_contrib

    if relative_HS_contribution is not None:
        req_H_sum = relative_HS_contribution[0] * error_metric_value
        req_S_sum = relative_HS_contribution[1] * error_metric_value
    else:
        current_H_contribution = total_H_sum / (total_H_sum + total_S_sum)
        req_H_sum = current_H_contribution * error_metric_value
        req_S_sum = (1 - current_H_contribution) * error_metric_value

    needed_H_free = req_H_sum - fixed_H_contrib
    needed_S_free = req_S_sum - fixed_S_contrib

    if 'H' in errorgen_types:
        if error_metric == 'generator_infidelity':
            H_scale_factor = _np.sqrt(needed_H_free / current_H_sum_free)
        if error_metric == 'total_generator_error':
            H_scale_factor = needed_H_free / current_H_sum_free
        for key in H_free_keys:
            random_rates_dicts['H'][key] *= H_scale_factor

    if 'S' in errorgen_types:
        S_scale_factor = needed_S_free / current_S_sum_free
        for key in S_free_keys:
            random_rates_dicts['S'][key] *= S_scale_factor
        for key in C_free_keys:
            random_rates_dicts['C'][key] *= S_scale_factor
        for key in A_free_keys:
            random_rates_dicts['A'][key] *= S_scale_factor

    return


def _convert_to_global_labels(
        errorgen_rates_dict: dict[_LEEL, float],
        state_space: _StateSpace,
        qubit_labels: list | None
    ) -> dict[_GEEL, float]:
    """Cast local error generator labels to global labels and apply any qubit label remapping."""
    result = {_GEEL.cast(lbl, sslbls=state_space.state_space_labels): val
              for lbl, val in errorgen_rates_dict.items()}
    if qubit_labels is not None:
        mapper = {i: lbl for i, lbl in enumerate(qubit_labels)}
        result = {lbl.map_state_space_labels(mapper): val for lbl, val in result.items()}
    return result


def random_CPTP_error_generator_rates(
        num_qubits        : int,
        errorgen_types    : tuple[Literal_HSCA, ...] = ('H', 'S', 'C', 'A'),
        max_weights       : dict[str, int] | None = None,
        H_params          : tuple[float, float] = (0., .01),
        SCA_params        : tuple[float, float] = (0., .01),
        error_metric      : Literal['generator_infidelity', 'total_generator_error'] | None = None,
        error_metric_value        : float | None = None,
        relative_HS_contribution  : tuple[float, float] | None = None,
        fixed_errorgen_rates      : dict[_LEEL, float] | None = None,
        sslbl_overlap             : list | None = None,
        label_type                : Literal['global', 'local'] = 'global',
        seed           : int | None = None,
        qubit_labels   : list | None = None
    ) -> dict:
    """
    Function for generating a random set of CPTP error generator rates.

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
        model. If specifying fixed C and A rates it is possible for the final
        error generator to be non-CP.

    label_type : str, optional (default 'global')
        String which can be either 'global' or 'local', indicating whether to
        return a dictionary with keys which are `GlobalElementaryErrorgenLabel`
        or `LocalElementaryErrorgenLabel` objects respectively.

    seed : int, optional (default None)
        An optional integer used in seeding the RNG.

    qubit_labels : list or int or str, optional (default None)
        An optional list of qubit labels upon which the error generator should act.
        Only utilized when returning global labels.

    Returns
    -------
    Dictionary of error generator coefficient labels and rates
    """
    if fixed_errorgen_rates is None:
        fixed_errorgen_rates = {}

    if label_type not in ['global', 'local']:
        raise ValueError('Unsupported label type {label_type}.')

    fixed_H_contrib, fixed_S_contrib = _validate_random_CPTP_params(
        errorgen_types, H_params, SCA_params, error_metric, error_metric_value,
        relative_HS_contribution, fixed_errorgen_rates, max_weights
    )
    rng = _np.random.default_rng(seed)

    state_space    = _QubitSpace.cast(num_qubits)
    errorgen_basis = _bo.CompleteElementaryErrorgenBasis('PP', state_space, errorgen_types, max_weights, sslbl_overlap, 'local')
    
    errgen_labels_H = _sort_errorgen_labels(errorgen_basis.sublabels('H'))
    errgen_labels_S = _sort_errorgen_labels(errorgen_basis.sublabels('S'))
    errgen_labels_C = _sort_errorgen_labels(errorgen_basis.sublabels('C'))
    errgen_labels_A = _sort_errorgen_labels(errorgen_basis.sublabels('A'))
    errgen_labels_C = _filter_ca_labels_for_cp(errgen_labels_C, errgen_labels_S)
    errgen_labels_A = _filter_ca_labels_for_cp(errgen_labels_A, errgen_labels_S)

    random_rates_dicts : dict[Literal_HSCA, dict[_LEEL, float]] = _generate_random_rates(
        errgen_labels_H, errgen_labels_S, errgen_labels_C, errgen_labels_A, H_params, SCA_params, rng
    )
    _check_ca_cp_constraint(random_rates_dicts['C'], random_rates_dicts['S'], 'C')
    _check_ca_cp_constraint(random_rates_dicts['A'], random_rates_dicts['S'], 'A')
    for egtyp, rate_dict in random_rates_dicts.items():
        for leel, rate in fixed_errorgen_rates.items():
            if leel.errorgen_type == egtyp:
                rate_dict[leel] = rate

    H_free_keys = [key for key in errgen_labels_H if key not in fixed_errorgen_rates]
    S_free_keys = [key for key in errgen_labels_S if key not in fixed_errorgen_rates]
    C_free_keys = [key for key in errgen_labels_C if key not in fixed_errorgen_rates]
    A_free_keys = [key for key in errgen_labels_A if key not in fixed_errorgen_rates]
    
    if error_metric is not None and error_metric_value is not None: 
        # ^ The `and` in that check is redundant, but keep it for type checking.
        _rescale_to_error_metric(
            random_rates_dicts, error_metric, error_metric_value,
            relative_HS_contribution, fixed_H_contrib, fixed_S_contrib,
            errorgen_types, H_free_keys, S_free_keys, C_free_keys, A_free_keys
        )

    errorgen_rates_dict = {}
    for errgen_type in errorgen_types:
        errorgen_rates_dict.update(random_rates_dicts[errgen_type])

    if label_type == 'global':
        errorgen_rates_dict = _convert_to_global_labels(errorgen_rates_dict, state_space, qubit_labels)

    return errorgen_rates_dict


def _sort_errorgen_labels(errgen_labels):
    """
    This function sorts error generator coefficients in canonical way.
    Helper function for random error generator rate construction. 
    """
    if not errgen_labels:
        return []
    
    assert isinstance(errgen_labels[0], _LEEL), 'Can only sort local labels at the moment'

    errorgen_types = [lbl.errorgen_type for lbl in errgen_labels]
    assert len(set(errorgen_types))==1, 'only one error generator type at a time is supported presently'

    errorgen_type = errorgen_types[0]
    if errorgen_type in ('H', 'S'):
        sorted_errgen_labels = sorted(errgen_labels, key= lambda lbl:lbl.basis_element_labels[0])
    else:
        sorted_errgen_labels = sorted(errgen_labels, key= lambda lbl:(lbl.basis_element_labels[0], lbl.basis_element_labels[1]))

    return sorted_errgen_labels

