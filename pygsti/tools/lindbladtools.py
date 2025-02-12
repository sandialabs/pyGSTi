"""
Utility functions relevant to Lindblad forms and projections
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.sparse as _sps

from pygsti.tools.basistools import basis_matrices
import pygsti.baseobjs as _bo
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel, \
                                          LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel
from pygsti.baseobjs.statespace import QubitSpace as _QubitSpace
import warnings as _warnings

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
def create_elementary_errorgen(typ, p, q=None, sparse=False):
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
    d = p.shape[0] 
    d2 = d**2
    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if typ in 'CA':
        pq_plus_qp = pdag @ q + qdag @ p
        pq_minus_qp = pdag @ q - qdag @ p

    #if p or q is a sparse matrix fall back to original implementation
    if not isinstance(p, _np.ndarray) or (q is not None and not isinstance(q, _np.ndarray)):
        # Loop through the standard basis as all possible input density matrices
        for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
            # Only difference between H/S/C/A is how they transform input density matrices
            if typ == 'H':
                rho1 = -1j * (p @ rho0 - rho0 @ p)  # Add "/2" to have PP ham gens match previous versions of pyGSTi
            elif typ == 'S':
                pdag_p = (pdag @ p)
                rho1 = p @ rho0 @ pdag - 0.5 * (pdag_p @ rho0 + rho0 @ pdag_p)
            elif typ == 'C':
                rho1 = p @ rho0 @ qdag + q @ rho0 @ pdag - 0.5 * (pq_plus_qp @ rho0 + rho0 @ pq_plus_qp)
            elif typ == 'A':
                rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag + 0.5 * (pq_minus_qp @ rho0 + rho0 @ pq_minus_qp))
            elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
    else:
        # Loop through the standard basis as all possible input density matrices
        for i in range(d): 
            for j in range(d):
                # Only difference between H/S/C/A is how they transform input density matrices
                if typ == 'H':
                    rho1 = _np.zeros((d,d), dtype=_np.complex128)
                    rho1[:, j] = -1j*p[:, i]
                    rho1[i, :] += 1j*p[j, :]
                elif typ == 'S':
                    pdag_p = (pdag @ p)
                    rho1 = p[:,i].reshape((d,1))@pdag[j,:].reshape((1,d))
                    rho1[:, j] += -.5*pdag_p[:, i]
                    rho1[i, :] += -.5*pdag_p[j, :]
                elif typ == 'C':
                    rho1 = p[:,i].reshape((d,1))@qdag[j,:].reshape((1,d)) + q[:,i].reshape((d,1))@pdag[j,:].reshape((1,d))
                    rho1[:, j] += -.5*pq_plus_qp[:, i]
                    rho1[i, :] += -.5*pq_plus_qp[j, :]
                elif typ == 'A':
                    rho1 = 1j*(p[:,i].reshape((d,1))@ qdag[j,:].reshape((1,d))) - 1j*(q[:,i].reshape((d,1))@pdag[j,:].reshape((1,d)))
                    rho1[:, j] += 1j*.5*pq_minus_qp[:, i]
                    rho1[i, :] += 1j*.5*pq_minus_qp[j, :]

                elem_errgen[:, d*i+j] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: elem_errgen = elem_errgen.tocsr()
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
                    rho1[:, j] += -.5*pq_plus_qp[:, i]
                    rho1[i, :] += -.5*pq_plus_qp[j, :]
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
        lind_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: lind_errgen = lind_errgen.tocsr()
    return lind_errgen


def random_error_generator_rates(num_qubits, errorgen_types=('H', 'S', 'C', 'A'), max_weights=None,
                                 H_params=(0.,.01), SCA_params=(0.,.01), error_metric=None, error_metric_value=None, 
                                 relative_HS_contribution=None, fixed_errorgen_rates=None, sslbl_overlap=None, 
                                 label_type='global', seed = None):
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
    
    Returns
    -------
    Dictionary of error generator coefficient labels and rates 
    """
    
    #Add various assertions
    if fixed_errorgen_rates is None:
        fixed_errorgen_rates = dict()
    
    if error_metric is not None:
        assert H_params[0] == 0. and SCA_params[0] == 0., 'Specifying non-zero HSCA means together with a target error metric is not supported.'
        if error_metric not in ('generator_infidelity', 'total_generator_error'):
            raise ValueError('Unsupported error metric type. Currently supported options are generator_infidelity and total_generator_error')
        #Add a check that the desired error metric value is attainable given the values of fixed_errorgen_rates.
        if fixed_errorgen_rates:
            #verify that all of the keys are LocalElementaryErrorgenLabel objects.
            msg = 'All keys of fixed_errorgen_rates must be LocalElementaryErrorgenLabel.'
            assert all([isinstance(key, _LocalElementaryErrorgenLabel) for key in fixed_errorgen_rates.keys()]), msg
            
            #get the H and S rates from the dictionary.
            fixed_H_rates = _np.array([val for key, val in fixed_errorgen_rates.items() if key.errorgen_type == 'H'])
            fixed_S_rates = _np.array([val for key, val in fixed_errorgen_rates.items() if key.errorgen_type == 'S'])
            fixed_S_contribution = _np.sum(fixed_S_rates)
            if error_metric == 'generator_infidelity':
                fixed_H_contribution = _np.sum(fixed_H_rates**2)
                fixed_error_metric_value = fixed_S_contribution + fixed_H_contribution
            elif error_metric == 'total_generator_error':
                fixed_H_contribution = _np.sum(_np.abs(fixed_H_rates))
                fixed_error_metric_value = fixed_S_contribution + fixed_H_contribution
            msg = f'Incompatible values of error_metric_value and fixed_errorgen_rates. The value of {error_metric}={error_metric_value}'\
                  + f' is less than the value of {fixed_error_metric_value} corresponding to the given fixed_errorgen_rates_dict.'
            assert fixed_error_metric_value < error_metric_value, msg
            
            if relative_HS_contribution is not None:
                msg_H = f'Fixed H contribution to {error_metric} of {fixed_H_contribution} exceeds overall H contribution target value of {relative_HS_contribution[0]*error_metric_value}.'
                msg_S = f'Fixed S contribution to {error_metric} of {fixed_S_contribution} exceeds overall S contribution target value of {relative_HS_contribution[1]*error_metric_value}.'
                assert fixed_H_contribution < relative_HS_contribution[0]*error_metric_value, msg_H
                assert fixed_S_contribution < relative_HS_contribution[1]*error_metric_value, msg_S
        else:
            fixed_H_contribution = 0
            fixed_S_contribution = 0
                      
    if relative_HS_contribution is not None:
        assert ('H' in errorgen_types and 'S' in errorgen_types), 'Invalid relative_HS_contribution, one of either H or S is not in errorgen_types.'
        if error_metric is None:
            _warnings.warn('The relative_HS_contribution kwarg is only utilized when error_metric is not None, the specified value is ignored otherwise.')
        else:
            assert abs(1-sum(relative_HS_contribution))<=1e-7, 'The relative_HS_contribution should sum to 1.'
        
    if max_weights is not None:
        assert max_weights['C'] <= max_weights['S'] and max_weights['A'] <= max_weights['S'], 'The maximum weight of the C and A terms should be less than or equal to the maximum weight of S.'
        assert max_weights['C'] == max_weights['A'], 'Maximum weight and C of A terms must be the same at present.'
    rng = _np.random.default_rng(seed)

    if 'C' in errorgen_types or 'A' in errorgen_types:
        assert 'C' in errorgen_types and 'A' in errorgen_types, 'Support only currently available for random C and A terms if both sectors present.'
        
    #create a state space with this dimension.
    state_space = _QubitSpace.cast(num_qubits)
    
    #create an error generator basis according the our weight specs
    errorgen_basis = _bo.CompleteElementaryErrorgenBasis('PP', state_space, elementary_errorgen_types=errorgen_types,
                                                         max_weights=max_weights, sslbl_overlap=sslbl_overlap, default_label_type='local')
    
    #Get the labels, broken out by sector, of each of the error generators in this basis.
    errgen_labels_H = _sort_errorgen_labels(errorgen_basis.sublabels('H'))
    errgen_labels_S = _sort_errorgen_labels(errorgen_basis.sublabels('S'))
    errgen_labels_C = _sort_errorgen_labels(errorgen_basis.sublabels('C'))
    errgen_labels_A = _sort_errorgen_labels(errorgen_basis.sublabels('A'))
    
    #filter out any C or A terms which can't be present with CP constraints due to lack of correct S term.
    filtered_errgen_labels_C = []
    for lbl in errgen_labels_C:
        first_label = _LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[0],))
        second_label = _LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[1],))
        if first_label not in errgen_labels_S or second_label not in errgen_labels_S:
            continue
        else:
            filtered_errgen_labels_C.append(lbl)
    filtered_errgen_labels_A = []
    for lbl in errgen_labels_A:
        first_label = _LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[0],))
        second_label = _LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[1],))
        if first_label not in errgen_labels_S or second_label not in errgen_labels_S:
            continue
        else:
            filtered_errgen_labels_A.append(lbl)
    errgen_labels_C = filtered_errgen_labels_C
    errgen_labels_A = filtered_errgen_labels_A

    #Get the number of H and S error generators. These are stored in HSCA order in the labels 
    num_H_rates = len(errgen_labels_H)
    num_S_rates = len(errgen_labels_S)
    
    random_rates_dicts = dict()
    #Generate random H rates
    random_rates_dicts['H'] = {lbl: val for lbl,val in zip(errgen_labels_H, rng.normal(loc=H_params[0], scale=H_params[1], size = num_H_rates))}
    
    #Create a random matrix with complex gaussian entries which will be used to generator a PSD matrix for the SCA rates.
    random_SCA_gen_mat = rng.normal(loc=SCA_params[0], scale=SCA_params[1], size=(num_S_rates, num_S_rates)) + \
                        1j* rng.normal(loc=SCA_params[0], scale=SCA_params[1], size=(num_S_rates, num_S_rates))
   
    random_SCA_mat = random_SCA_gen_mat @ random_SCA_gen_mat.conj().T
    #The random S rates are just the diagonal of random_SCA_mat.
    random_rates_dicts['S'] = {lbl: val for lbl,val in zip(errgen_labels_S,  _np.real(_np.diag(random_SCA_mat)).copy())}
    #The random C rates are the real part of the off diagonal entries, and the A rates the imaginary part.
    random_rates_dicts['C'] =  {lbl: val for lbl,val in zip(errgen_labels_C, random_SCA_mat[_np.triu_indices_from(random_SCA_mat, k=1)].real)}
    random_rates_dicts['A'] =  {lbl: val for lbl,val in zip(errgen_labels_A, random_SCA_mat[_np.triu_indices_from(random_SCA_mat, k=1)].imag)}
    #manually check conditions on C and A
    for lbl, rate in random_rates_dicts['C'].items():
        first_S_rate = random_rates_dicts['S'][_LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[0],))]
        second_S_rate = random_rates_dicts['S'][_LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[1],))]

        if not (abs(rate) <= _np.sqrt(first_S_rate*second_S_rate)):
            print(f'{lbl}: {rate}')
            raise ValueError('Invalid C rate')
        
    #manually check conditions on C and A
    for lbl, rate in random_rates_dicts['A'].items():
        first_S_rate = random_rates_dicts['S'][_LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[0],))]
        second_S_rate = random_rates_dicts['S'][_LocalElementaryErrorgenLabel('S', (lbl.basis_element_labels[1],))]

        if not (abs(rate) <= _np.sqrt(first_S_rate*second_S_rate)):
            print(f'{lbl}: {rate}')
            raise ValueError('Invalid A rate')

    #Add in/override the fixed rates for each of the sectors.
    H_fixed_keys = []
    S_fixed_keys = []
    C_fixed_keys = []
    A_fixed_keys = []
    for key in fixed_errorgen_rates:
        if key.errorgen_type == 'H':
            H_fixed_keys.append(key)
        elif key.errorgen_type == 'S':
            S_fixed_keys.append(key)
        elif key.errorgen_type == 'C':
            C_fixed_keys.append(key)
        else:
            A_fixed_keys.append(key)
          
    random_rates_dicts['H'].update({key:fixed_errorgen_rates[key] for key in H_fixed_keys})
    random_rates_dicts['S'].update({key:fixed_errorgen_rates[key] for key in S_fixed_keys})
    random_rates_dicts['C'].update({key:fixed_errorgen_rates[key] for key in C_fixed_keys})
    random_rates_dicts['A'].update({key:fixed_errorgen_rates[key] for key in A_fixed_keys})
    
    #For each sector construct a complementary structure of the free(ish) parameters error generator parameters for
    #that sector.
    H_free_keys = [key for key in errgen_labels_H if key not in fixed_errorgen_rates] #membership checking is (often) faster with dicts
    S_free_keys = [key for key in errgen_labels_S if key not in fixed_errorgen_rates]
    C_free_keys = [key for key in errgen_labels_C if key not in fixed_errorgen_rates]
    A_free_keys = [key for key in errgen_labels_A if key not in fixed_errorgen_rates]
    
    #Now it is time to apply the various normalizations necessary to get the desired target
    #generator infidelity and sector weights.
    if error_metric is not None:
        #Get the free parameter's  For both generator infidelity we use the sum of the S rates
        current_S_sum_free = _np.sum([random_rates_dicts['S'][key] for key in S_free_keys])
        if error_metric == 'generator_infidelity':
            #for generator infidelity we use the sum of the squared H rates.
            current_H_sum_free = _np.sum([random_rates_dicts['H'][key]**2 for key in H_free_keys])
        elif error_metric == 'total_generator_error':
            #for total generator error we use the sum of the H rates directly.
            current_H_sum_free = _np.sum([abs(random_rates_dicts['H'][key]) for key in H_free_keys])
        
        total_H_sum = current_H_sum_free + fixed_H_contribution
        total_S_sum = current_S_sum_free + fixed_S_contribution
        
        if relative_HS_contribution is not None:
            #calculate the target values of the H and S contributions to the error metric 
            #given the specified contributions
            req_H_sum = relative_HS_contribution[0]*error_metric_value
            req_S_sum = relative_HS_contribution[1]*error_metric_value
                        
        #If we haven't specified a relative contribution for H and S then we will scale these
        #to give the correct generator infidelity while preserving whatever relative contribution
        #to the generator infidelity they were randomly sampled to have.
        else:
            #Get the current relative contributions.
            current_H_contribution = total_H_sum/(total_H_sum+total_S_sum)
            current_S_contribution = 1-current_H_contribution
            req_H_sum = current_H_contribution*error_metric_value
            req_S_sum = current_S_contribution*error_metric_value
            
        #this is how much we still need to be contributed by the free parameters
        needed_H_free = req_H_sum - fixed_H_contribution
        needed_S_free = req_S_sum - fixed_S_contribution
        
        if error_metric == 'generator_infidelity':
            #The scale factor for the H rates is sqrt(req_squared_H_sum/current_squared_H_sum)
            H_scale_factor = _np.sqrt(needed_H_free/current_H_sum_free)
        elif error_metric == 'total_generator_error':
            #The scale factor for the S rates is req_S_sum/current_S_sum
            H_scale_factor = needed_H_free/current_H_sum_free    
        #The scale factor for the S rates is req_S_sum/current_S_sum
        S_scale_factor = needed_S_free/current_S_sum_free    

        #Rescale the free random rates, note that the free SCA terms will all be scaled by the S_scale_factor
        #to preserve PSD.
        for key in H_free_keys:
            random_rates_dicts['H'][key]*=H_scale_factor
        for key in S_free_keys:
            random_rates_dicts['S'][key]*=S_scale_factor
        for key in C_free_keys:
            random_rates_dicts['C'][key]*=S_scale_factor
        for key in A_free_keys:
            random_rates_dicts['A'][key]*=S_scale_factor
                
    #Now turn this into a rates dict
    errorgen_rates_dict = dict()
    for errgen_type in errorgen_types:
        errorgen_rates_dict.update(random_rates_dicts[errgen_type])
    
    if label_type not in ['global', 'local']:
        raise ValueError('Unsupported label type {label_type}.')

    if label_type == 'global':
        errorgen_rates_dict = {_GlobalElementaryErrorgenLabel.cast(lbl, sslbls=state_space.state_space_labels): val 
                               for lbl, val in  errorgen_rates_dict.items()}

    return errorgen_rates_dict

def _sort_errorgen_labels(errgen_labels):
    """
    This function sorts error generator coefficients in canonical way.
    Helper function for random error generator rate construction. 
    """
    if not errgen_labels:
        return []
    
    assert isinstance(errgen_labels[0], _LocalElementaryErrorgenLabel), 'Can only sort local labels at the moment'

    errorgen_types = [lbl.errorgen_type for lbl in errgen_labels]
    assert len(set(errorgen_types))==1, 'only one error generator type at a time is supported presently'

    errorgen_type = errorgen_types[0]
    if errorgen_type in ('H', 'S'):
        sorted_errgen_labels = sorted(errgen_labels, key= lambda lbl:lbl.basis_element_labels[0])
    else:
        sorted_errgen_labels = sorted(errgen_labels, key= lambda lbl:(lbl.basis_element_labels[0], lbl.basis_element_labels[1]))

    return sorted_errgen_labels

