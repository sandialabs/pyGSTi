""" Utility functions relevant to Lindblad forms and projections """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy.sparse as _sps
from ..baseobjs.basis import basis_matrices
from . import matrixtools as _mt

def hamiltonian_to_lindbladian(hamiltonian, sparse=False):
    """
    Construct the Lindbladian corresponding to a given Hamiltonian.

    Mathematically, for a d-dimensional Hamiltonian matrix H, this
    routine constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = -1j*[ H, rho ], where square brackets
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
        lindbladian = _sps.lil_matrix( (d**2,d**2), dtype=hamiltonian.dtype )
    else:
        lindbladian = _np.empty( (d**2,d**2), dtype=hamiltonian.dtype )

    for i,rho0 in enumerate(basis_matrices('std',d)): #rho0 == input density mx
        rho1 = -1j*(_mt.safedot(hamiltonian,rho0) - _mt.safedot(rho0,hamiltonian))
        lindbladian[:,i] = rho1.flatten()[:,None] if sparse else rho1.flatten()
          # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian



def stochastic_lindbladian(Q, sparse=False):
    """
    Construct the Lindbladian corresponding to stochastic Q-errors.

    Mathematically, for a d-dimensional matrix Q, this routine 
    constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = Q*rho*Q^dag where rho is a density
    matrix.  L is returned as a superoperator matrix that acts on a
    vectorized density matrices.

    Parameters
    ----------
    Q : ndarray
      The matrix used to construct the Lindbladian.

    sparse : bool, optional
      Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Q.shape) == 2)
    assert(Q.shape[0] == Q.shape[1])
    Qdag = _np.conjugate(_np.transpose(Q))
    d = Q.shape[0]
    if sparse:
        lindbladian = _sps.lil_matrix( (d**2,d**2), dtype=Q.dtype )
    else:
        lindbladian = _np.empty( (d**2,d**2), dtype=Q.dtype )

    for i,rho0 in enumerate(basis_matrices('std',d)): #rho0 == input density mx
        rho1 = _mt.safedot(Q,_mt.safedot(rho0,Qdag))
        lindbladian[:,i] = rho1.flatten()[:,None] if sparse else rho1.flatten()
          # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


def affine_lindbladian(Q, sparse=False):
    """
    Construct the Lindbladian corresponding to affine Q-errors.

    Mathematically, for a d-dimensional matrix Q, this routine 
    constructs the d^2-dimension Lindbladian matrix L whose
    action is given by L(rho) = Q where rho is a density
    matrix.  L is returned as a superoperator matrix that acts on a
    vectorized density matrices.

    Parameters
    ----------
    Q : ndarray
      The matrix used to construct the Lindbladian.

    sparse : bool, optional
      Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Q.shape) == 2)
    assert(Q.shape[0] == Q.shape[1])
    d = Q.shape[0]
    Id = _np.identity(d,'d').flatten()
    if sparse:
        lindbladian = _sps.lil_matrix( (d**2,d**2), dtype=Q.dtype )
    else:
        lindbladian = _np.empty( (d**2,d**2), dtype=Q.dtype )

    for i,rho0 in enumerate(basis_matrices('std',d)): #rho0 == input density mx
        rho1 = Q * _mt.safedot(Id,rho0.flatten()) # get |Q>><Id|rho0
        lindbladian[:,i] = rho1.flatten()[:,None] if sparse else rho1.flatten()
          # vectorize rho1 & set as linbladian column

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


def nonham_lindbladian(Lm,Ln,sparse=False):
    """
    Construct the Lindbladian corresponding to generalized
    non-Hamiltonian (stochastic) errors.

    Mathematically, for d-dimensional matrices Lm and Ln, this routine 
    constructs the d^2-dimension Lindbladian matrix L whose action is
    given by:

    L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)

    where rho is a density matrix.  L is returned as a superoperator
    matrix that acts on a vectorized density matrices.

    Parameters
    ----------
    Lm, Ln : ndarray
      The matrices used to construct the Lindbladian.

    sparse : bool, optional
      Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Lm.shape) == 2)
    assert(Lm.shape[0] == Lm.shape[1])
    Lm_dag = _np.conjugate(_np.transpose(Lm))
    d = Lm.shape[0]
    if sparse:
        lindbladian = _sps.lil_matrix( (d**2,d**2), dtype=Lm.dtype )
    else:
        lindbladian = _np.empty( (d**2,d**2), dtype=Lm.dtype )
        
#    print("BEGIN VERBOSE") #DEBUG!!!
    for i,rho0 in enumerate(basis_matrices('std',d)): #rho0 == input density mx
        rho1 = _mt.safedot(Ln,_mt.safedot(rho0,Lm_dag)) - 0.5 * (
            _mt.safedot(rho0,_mt.safedot(Lm_dag,Ln))+_mt.safedot(_mt.safedot(Lm_dag,Ln),rho0))
#        print("rho0[%d] = \n" % i,rho0)
#        print("rho1[%d] = \n" % i,rho1)
        lindbladian[:,i] = rho1.flatten()[:,None] if sparse else rho1.flatten()
          # vectorize rho1 & set as linbladian column
#    print("FINAL = \n",lindbladian)
#    print("END VERBOSE\n")

    if sparse: lindbladian = lindbladian.tocsr()
    return lindbladian


