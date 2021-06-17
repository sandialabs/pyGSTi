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


def hamiltonian_to_lindbladian(hamiltonian, sparse=False):
    """
    Construct the Lindbladian corresponding to a given Hamiltonian.

    Mathematically, for a d-dimensional Hamiltonian matrix H, this
    routine constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = -1j*sqrt(d)/2*[ H, rho ], where square brackets
    denote the commutator and rho is a density matrix.  L is returned
    as a superoperator matrix that acts on a vectorized density matrices.

    Parameters
    ----------
    hamiltonian : ndarray
        The hamiltonian matrix used to construct the Lindbladian.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(hamiltonian.shape) == 2)
    assert(hamiltonian.shape[0] == hamiltonian.shape[1])
    d = hamiltonian.shape[0]
    if sparse:
        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=hamiltonian.dtype)
    else:
        lindbladian = _np.empty((d**2, d**2), dtype=hamiltonian.dtype)

    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
        rho1 = _np.sqrt(d) / 2 * (-1j * (_mt.safe_dot(hamiltonian, rho0) - _mt.safe_dot(rho0, hamiltonian)))
        lindbladian[:, i] = _np.real_if_close(rho1.flatten()[:, None] if sparse else rho1.flatten())
        # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


def stochastic_lindbladian(q, sparse=False):
    """
    Construct the Lindbladian corresponding to stochastic q-errors.

    Mathematically, for a d-dimensional matrix q, this routine
    constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = q*rho*q^dag where rho is a density
    matrix.  L is returned as a superoperator matrix that acts on a
    vectorized density matrices.

    Parameters
    ----------
    q : ndarray
        The matrix used to construct the Lindbladian.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    # single element basis (plus identity)
    # if lambda is coefficient of stochastic term using normalized basis els, then
    # exp(-d*lambda) == pault-transfer-mx diag = 1 - d^2*err_rate
    # so lambda = -log(1-d^2*err_rate) / d (where err_rate is the per-Pauli stochastic err rate)
    # if lambda is coefficient using normalized * sqrt(d) (e.g. un-normalized Pauli ops)
    # then exp(-d^2*lambda) = pault-transfer-mx diag so lambda = -log(1-d^2*err_rate) / d^2
    # and since log(1+x) ~ x, lambda ~= d^2*err_rate) / d^2 = err_rate.
    #This is the most intuitive to the user (the coeff lambda ~= err_rate), so we
    # scale the generator to by a sqrt(d) factor per basis element, as
    # we expect the given element q to be normalized.

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(q.shape) == 2)
    assert(q.shape[0] == q.shape[1])
    Qdag = _np.conjugate(_np.transpose(q))
    d = q.shape[0]
    if sparse:
        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=q.dtype)
    else:
        lindbladian = _np.empty((d**2, d**2), dtype=q.dtype)

    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
        rho1 = d * _mt.safe_dot(q, _mt.safe_dot(rho0, Qdag))
        lindbladian[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


def affine_lindbladian(q, sparse=False):
    """
    Construct the Lindbladian corresponding to affine q-errors.

    Mathematically, for a d-dimensional matrix q, this routine
    constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = q where rho is a density
    matrix.  L is returned as a superoperator matrix that acts on a
    vectorized density matrices.

    Parameters
    ----------
    q : ndarray
        The matrix used to construct the Lindbladian.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(q.shape) == 2)
    assert(q.shape[0] == q.shape[1])
    d = q.shape[0]
    Id = _np.identity(d, 'd').flatten()
    if sparse:
        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=q.dtype)
    else:
        lindbladian = _np.empty((d**2, d**2), dtype=q.dtype)

    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
        rho1 = q * _mt.safe_dot(Id, rho0.flatten())  # get |q>><Id|rho0
        lindbladian[:, i] = rho1.to_dense(on_space='HilbertSchmidt').flatten().T \
            if sparse else rho1.flatten()  # weird that need .T here
        # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


def nonham_lindbladian(Lm, Ln, sparse=False):  # noqa N803
    """
    Construct the Lindbladian corresponding to generalized non-Hamiltonian (stochastic) errors.

    Mathematically, for d-dimensional matrices Lm and Ln, this routine
    constructs the d^2-dimension Lindbladian matrix L whose action is
    given by:

    L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)

    where rho is a density matrix.  L is returned as a superoperator
    matrix that acts on a vectorized density matrices.

    Parameters
    ----------
    Lm : numpy.ndarray
        d-dimensional matrix.

    Ln : numpy.ndarray
        d-dimensional matrix.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    #Same sqrt(d) per basis element (so total d) scaling factor as
    # stochastic_lindbladian (see notes there).

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Lm.shape) == 2)
    assert(Lm.shape[0] == Lm.shape[1])
    Lm_dag = _np.conjugate(_np.transpose(Lm))
    d = Lm.shape[0]
    if sparse:
        lindbladian = _sps.lil_matrix((d**2, d**2), dtype=Lm.dtype)
    else:
        lindbladian = _np.empty((d**2, d**2), dtype=Lm.dtype)

#    print("BEGIN VERBOSE") #DEBUG!!!
    for i, rho0 in enumerate(basis_matrices('std', d**2)):  # rho0 == input density mx
        rho1 = _mt.safe_dot(Ln, _mt.safe_dot(rho0, Lm_dag)) - 0.5 * (
            _mt.safe_dot(rho0, _mt.safe_dot(Lm_dag, Ln)) + _mt.safe_dot(_mt.safe_dot(Lm_dag, Ln), rho0))
        rho1 *= d
#        print("rho0[%d] = \n" % i,rho0)
#        print("rho1[%d] = \n" % i,rho1)
        lindbladian[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()
        # vectorize rho1 & set as linbladian column
#    print("FINAL = \n",lindbladian)
#    print("END VERBOSE\n")

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian
