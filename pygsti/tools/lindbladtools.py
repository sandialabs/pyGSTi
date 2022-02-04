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

from pygsti.tools import matrixtools as _mt
from pygsti.tools.basistools import basis_matrices


#def _jamiol_unit_norm(P, Q=None):
#    d = P.shape[0]
#    L = _np.zeros((d**2, d**2), dtype=P.dtype)
#
#    # Loop through the standard basis as all possible input density matrices
#    rhos = basis_matrices('std', d**2)
#    for i, rho0 in enumerate(rhos):
#        if Q is None:
#            rho1 = P @ rho0
#        else:
#            rho1 = P @ rho0 @ Q
#        L[:, i] = rho1.flatten()
#    flatL = L.flatten()
#    return _np.vdot(flatL, flatL)     

def create_elementary_errorgen_dual(typ, p, q=None, sparse=False, normalization_factor='auto'):
    """
    Construct a "dual" elementary error generator matrix in the "standard" (matrix-unit) basis.

    The elementary error generator that is dual to the one computed by calling
    :function:`create_elementary_errorgen` with the same argument.  This dual element
    can be used to find the coefficient of the original, or "primal" elementary generator.
    For example, if `A = sum(c_i * E_i)`, where `E_i` are the elementary error generators given
    by :function:`create_elementary_errorgen`), then `c_i = dot(D_i.conj(), A)` where `D_i`
    is the dual to `E_i`.

    There are four different types of dual elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j/(2d^2) * [ p, rho ]`
    Stochastic:   `L(rho) = 1/(d^2) p * rho * p
    Correlation:  `L(rho) = 1/(2d^2) ( p * rho * q + q * rho * p)
    Active:       `L(rho) = 1j/(2d^2) ( p * rho * q - q * rho * p)

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
    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    # Loop through the standard basis as all possible input density matrices
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

    return_normalization = bool(normalization_factor == 'auto_return')
    if normalization_factor in ('auto', 'auto_return'):
        primal = create_elementary_errorgen(typ, p, q, sparse)
        if sparse:
            normalization_factor = _np.vdot(elem_errgen.toarray().flatten(), primal.toarray().flatten())
        else:
            normalization_factor = _np.vdot(elem_errgen.flatten(), primal.flatten())
    elem_errgen *= _np.asscalar(_np.real_if_close(1 / normalization_factor))

    if sparse: elem_errgen = elem_errgen.tocsr()
    return (elem_errgen, normalization_factor) if return_normalization else elem_errgen


def create_elementary_errorgen(typ, p, q=None, sparse=False):
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.

    There are four different types of elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j * [ p, rho ]`
    Stochastic:   `L(rho) = p * rho * p - rho
    Correlation:  `L(rho) = p * rho * q + q * rho * p - 0.5 {{p,q}, rho}
    Active:       `L(rho) = 1j( p * rho * q - q * rho * p + 0.5 {[p,q], rho} )

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
    d = p.shape[0]; d2 = d**2
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

    # Loop through the standard basis as all possible input density matrices
    for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
        # Only difference between H/S/C/A is how they transform input density matrices
        if typ == 'H':
            rho1 = -1j * (p @ rho0 - rho0 @ p)
        elif typ == 'S':
            pdag_p = (pdag @ p)
            rho1 = p @ rho0 @ pdag - 0.5 * (pdag_p @ rho0 + rho0 @ pdag_p)
        elif typ == 'C':
            rho1 = p @ rho0 @ qdag + q @ rho0 @ pdag - 0.5 * (pq_plus_qp @ rho0 + rho0 @ pq_plus_qp)
        elif typ == 'A':
            rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag + 0.5 * (pq_minus_qp @ rho0 + rho0 @ pq_minus_qp))

        elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: elem_errgen = elem_errgen.tocsr()
    return elem_errgen

def create_lindbladian_term_errorgen(typ, Lm, Ln=None, sparse=False):  # noqa N803
    """
    Construct the superoperator for a term in the common Lindbladian expansion of an error generator.

    Mathematically, for d-dimensional matrices Lm and Ln, this routine
    constructs the d^2-dimension Lindbladian matrix L whose action is
    given by:

    L(rho) = -i [Lm, rho]     (when `typ == 'H'`)

    or

    L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)    (`typ == 'O'`)

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





#REMOVE
#def hamiltonian_to_lindbladian(hamiltonian, sparse=False):
#    """
#    Construct the Lindbladian corresponding to a given Hamiltonian.
#
#    Mathematically, for a d-dimensional Hamiltonian matrix H, this
#    routine constructs the d^2-dimension Lindbladian matrix L whose
#    action is given by L(rho) = -1j*sqrt(d)/2*[ H, rho ], where square brackets
#    denote the commutator and rho is a density matrix.  L is returned
#    as a superoperator matrix that acts on a vectorized density matrices.
#
#    Parameters
#    ----------
#    hamiltonian : ndarray
#        The hamiltonian matrix used to construct the Lindbladian.
#
#    sparse : bool, optional
#        Whether to construct a sparse or dense (the default) matrix.
#
#    Returns
#    -------
#    ndarray or Scipy CSR matrix
#    """
#
#    #TODO: there's probably a fast & slick way to so this computation
#    #  using vectorization identities
#    assert(len(hamiltonian.shape) == 2)
#    assert(hamiltonian.shape[0] == hamiltonian.shape[1])
#    d = hamiltonian.shape[0]
#    if sparse:
#        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=hamiltonian.dtype)
#    else:
#        lindbladian = _np.empty((d**2, d**2), dtype=hamiltonian.dtype)
#
#    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
#        rho1 = _np.sqrt(d) / 2 * (-1j * (_mt.safe_dot(hamiltonian, rho0) - _mt.safe_dot(rho0, hamiltonian)))
#        lindbladian[:, i] = _np.real_if_close(rho1.flatten()[:, None] if sparse else rho1.flatten())
#        # vectorize rho1 & set as linbladian column
#
#    if sparse: lindbladian = lindbladian.tocsr()
#    return lindbladian
#
#
#def stochastic_lindbladian(q, sparse=False):
#    """
#    Construct the Lindbladian corresponding to stochastic q-errors.
#
#    Mathematically, for a d-dimensional matrix q, this routine
#    constructs the d^2-dimension Lindbladian matrix L whose
#    action is given by L(rho) = q*rho*q^dag where rho is a density
#    matrix.  L is returned as a superoperator matrix that acts on a
#    vectorized density matrices.
#
#    Parameters
#    ----------
#    q : ndarray
#        The matrix used to construct the Lindbladian.
#
#    sparse : bool, optional
#        Whether to construct a sparse or dense (the default) matrix.
#
#    Returns
#    -------
#    ndarray or Scipy CSR matrix
#    """
#
#    # single element basis (plus identity)
#    # if lambda is coefficient of stochastic term using normalized basis els, then
#    # exp(-d*lambda) == pault-transfer-mx diag = 1 - d^2*err_rate
#    # so lambda = -log(1-d^2*err_rate) / d (where err_rate is the per-Pauli stochastic err rate)
#    # if lambda is coefficient using normalized * sqrt(d) (e.g. un-normalized Pauli ops)
#    # then exp(-d^2*lambda) = pault-transfer-mx diag so lambda = -log(1-d^2*err_rate) / d^2
#    # and since log(1+x) ~ x, lambda ~= d^2*err_rate) / d^2 = err_rate.
#    #This is the most intuitive to the user (the coeff lambda ~= err_rate), so we
#    # scale the generator to by a sqrt(d) factor per basis element, as
#    # we expect the given element q to be normalized.
#
#    #TODO: there's probably a fast & slick way to so this computation
#    #  using vectorization identities
#    assert(len(q.shape) == 2)
#    assert(q.shape[0] == q.shape[1])
#    Qdag = _np.conjugate(_np.transpose(q))
#    d = q.shape[0]
#    if sparse:
#        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=q.dtype)
#    else:
#        lindbladian = _np.empty((d**2, d**2), dtype=q.dtype)
#
#    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
#        rho1 = d * _mt.safe_dot(q, _mt.safe_dot(rho0, Qdag))
#        lindbladian[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
#        # vectorize rho1 & set as linbladian column
#
#    if sparse: lindbladian = lindbladian.tocsr()
#    return lindbladian
#
#
#def affine_lindbladian(q, sparse=False):
#    """
#    Construct the Lindbladian corresponding to affine q-errors.
#
#    Mathematically, for a d-dimensional matrix q, this routine
#    constructs the d^2-dimension Lindbladian matrix L whose
#    action is given by L(rho) = q where rho is a density
#    matrix.  L is returned as a superoperator matrix that acts on a
#    vectorized density matrices.
#
#    Parameters
#    ----------
#    q : ndarray
#        The matrix used to construct the Lindbladian.
#
#    sparse : bool, optional
#        Whether to construct a sparse or dense (the default) matrix.
#
#    Returns
#    -------
#    ndarray or Scipy CSR matrix
#    """
#
#    #TODO: there's probably a fast & slick way to so this computation
#    #  using vectorization identities
#    assert(len(q.shape) == 2)
#    assert(q.shape[0] == q.shape[1])
#    d = q.shape[0]
#    Id = _np.identity(d, 'd').flatten()
#    if sparse:
#        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=q.dtype)
#    else:
#        lindbladian = _np.empty((d**2, d**2), dtype=q.dtype)
#
#    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
#        rho1 = q * _mt.safe_dot(Id, rho0.flatten())  # get |q>><Id|rho0
#        lindbladian[:, i] = rho1.todense().flatten().T \
#            if sparse else rho1.flatten()  # weird that need .T here
#        # vectorize rho1 & set as linbladian column
#
#    if sparse: lindbladian = lindbladian.tocsr()
#    return lindbladian
#
#
#def nonham_lindbladian(Lm, Ln, sparse=False):  # noqa N803
#    """
#    Construct the Lindbladian corresponding to generalized non-Hamiltonian (stochastic) errors.
#
#    Mathematically, for d-dimensional matrices Lm and Ln, this routine
#    constructs the d^2-dimension Lindbladian matrix L whose action is
#    given by:
#
#    L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)
#
#    where rho is a density matrix.  L is returned as a superoperator
#    matrix that acts on a vectorized density matrices.
#
#    Parameters
#    ----------
#    Lm : numpy.ndarray
#        d-dimensional matrix.
#
#    Ln : numpy.ndarray
#        d-dimensional matrix.
#
#    sparse : bool, optional
#        Whether to construct a sparse or dense (the default) matrix.
#
#    Returns
#    -------
#    ndarray or Scipy CSR matrix
#    """
#    #Same sqrt(d) per basis element (so total d) scaling factor as
#    # stochastic_lindbladian (see notes there).
#
#    #TODO: there's probably a fast & slick way to so this computation
#    #  using vectorization identities
#
#    #For performance, use dense ops when they're small, even when sparse=True
#    d = Lm.shape[0]
#    build_sparse_mx = sparse and d > 4  # for < 2Q matrices just convert to sparse at end (it's faster)
#    if build_sparse_mx:
#        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=Lm.dtype)
#    else:
#        lindbladian = _np.empty((d**2, d**2), dtype=Lm.dtype)
#        if _sps.issparse(Lm): Lm = Lm.toarray()
#        if _sps.issparse(Ln): Ln = Ln.toarray()
#
#    assert(len(Lm.shape) == 2)
#    assert(Lm.shape[0] == Lm.shape[1])
#    Lm_dag = _np.conjugate(_np.transpose(Lm))
#
#    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
#        rho1 = _mt.safe_dot(Ln, _mt.safe_dot(rho0, Lm_dag)) - 0.5 * (
#            _mt.safe_dot(rho0, _mt.safe_dot(Lm_dag, Ln)) + _mt.safe_dot(_mt.safe_dot(Lm_dag, Ln), rho0))
#        rho1 *= d
#        # print("rho0[%d] = \n" % i,rho0)
#        # print("rho1[%d] = \n" % i,rho1)
#        lindbladian[:, i] = rho1.flatten()[:, None] if build_sparse_mx else rho1.flatten()
#
#    if sparse:
#        lindbladian = lindbladian.tocsr() if build_sparse_mx else _sps.csr_matrix(lindbladian)
#    return lindbladian
