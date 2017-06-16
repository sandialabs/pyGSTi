from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from .basistools import *

def hamiltonian_to_lindbladian(hamiltonian):
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

    Returns
    -------
    ndarray
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(hamiltonian.shape) == 2)
    assert(hamiltonian.shape[0] == hamiltonian.shape[1])
    d = hamiltonian.shape[0]
    lindbladian = _np.empty( (d**2,d**2), dtype=hamiltonian.dtype )

    for i,rho0 in enumerate(std_matrices(d)): #rho0 == input density mx
        rho1 = -1j*(_np.dot(hamiltonian,rho0) - _np.dot(rho0,hamiltonian))
        lindbladian[:,i] = rho1.flatten()
          # vectorize rho1 & set as linbladian column

    return lindbladian



def stochastic_lindbladian(Q):
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

    Returns
    -------
    ndarray
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Q.shape) == 2)
    assert(Q.shape[0] == Q.shape[1])
    Qdag = _np.conjugate(_np.transpose(Q))
    d = Q.shape[0]
    lindbladian = _np.empty( (d**2,d**2), dtype=Q.dtype )

    for i,rho0 in enumerate(std_matrices(d)): #rho0 == input density mx
        rho1 = _np.dot(Q,_np.dot(rho0,Qdag))
        lindbladian[:,i] = rho1.flatten()
          # vectorize rho1 & set as linbladian column

    return lindbladian


def nonham_lindbladian(Lm,Ln):
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

    Returns
    -------
    ndarray
    """

    #TODO: there's probably a fast & slick way to so this computation
    #  using vectorization identities
    assert(len(Lm.shape) == 2)
    assert(Lm.shape[0] == Lm.shape[1])
    Lm_dag = _np.conjugate(_np.transpose(Lm))
    d = Lm.shape[0]
    lindbladian = _np.empty( (d**2,d**2), dtype=Lm.dtype )

#    print("BEGIN VERBOSE") #DEBUG!!!
    for i,rho0 in enumerate(std_matrices(d)): #rho0 == input density mx
        rho1 = _np.dot(Ln,_np.dot(rho0,Lm_dag)) - 0.5 * (
            _np.dot(rho0,_np.dot(Lm_dag,Ln))+_np.dot(_np.dot(Lm_dag,Ln),rho0))
#        print("rho0[%d] = \n" % i,rho0)
#        print("rho1[%d] = \n" % i,rho1)
        lindbladian[:,i] = rho1.flatten()
          # vectorize rho1 & set as linbladian column
#    print("FINAL = \n",lindbladian)
#    print("END VERBOSE\n")

    return lindbladian


