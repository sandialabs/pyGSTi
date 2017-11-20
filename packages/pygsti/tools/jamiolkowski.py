"""Utility functions related to the Choi representation of gates."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from ..baseobjs.basis import basis_matrices as _basis_matrices
from . import basistools as _bt
from . import matrixtools as _mt


# Gate Mx G:      rho  --> G rho                    where G and rho are in the Pauli basis (by definition/convention)
#            vec(rhoS) --> GStd vec(rhoS)           where GS and rhoS are in the std basis, GS = PtoS * G * StoP
# Choi Mx J:     rho  --> sum_ij Jij Bi rho Bj^dag  where Bi is some basis of mxs for rho-space; independent of basis for rho and Bi
#           vec(rhoS) --> sum_ij Jij (BSi x BSj^*) vec(rhoS)  where rhoS and BSi's are in std basis
#  Now,
#       Jkl = Trace( sum_ij Jij (BSi x BSj^*) , (BSk x BSl^*)^dag ) / Trace( (BSk x BSl^*), (BSk x BSl^*)^dag )
#           = Trace( GStd , (BSk x BSl^*)^dag ) / Trace( (BSk x BSl^*), (BSk x BSl^*)^dag )
#  In below function, take Bi's to be Pauli matrices
#  Note: vec(.) vectorization above is assumed to be done by-*rows* (as numpy.flatten does).

#Note that in just the std basis, the construction of the Jamiolkowski representation of a process phi is
#  J(Phi) = sum_(0<i,j<n) Phi(|i><j|) x |i><j|    where {|i>}_1^n spans the state space
#
#  Derivation: if we write:
#    Phi(|i><j|) = sum_kl C[(kl)(ij)] |k><l|
#  and
#    rho = sum_ij rho_ij |i><j|
#  then
#    Phi(rho) = sum_(ij)(kl) C[(kl)(ij)] rho_ij |k><l|
#             = sum_(ij)(kl) C[(kl)(ij)] |k> rho_ij <l|
#             = sum_(ij)(kl) C[(kl)(ij)] |k> <i| rho |j> <l|
#             = sum_(ij)(kl) C[(ik)(jl)] |i> <j| rho |l> <k|  (just permute index labels)
#  The definition of the Jamiolkoski matrix J is:
#    Phi(rho) = sum_(ij)(kl) J(ij)(kl) |i><j| rho |l><k|
#  so
#    J(ij)(kl) == C[(ik)(jl)]
#
#  Note: |i><j| x |k><l| is an object in "gate/process" space, since
#    it maps a vectorized density matrix, e.g. |a><b| to another density matrix via:
#    (|i><j| x |k><l|) vec(|a><b|) = [mx with 1 in (i*dmDim + k) row and (j*dmDim + l) col][vec with 1 in a*dmDim+b row]
#                                  = vec(|i><k|) if |a><b| == |j><l| else 0
#    so (|i><j| x |k><l|) vec(|j><l|) = vec(|i><k|)
#    and could write as: (|ik><jl|) |jl> = |ik>
#
# Now write J as:
#    J  = sum_ijkl |ij> J(ij)(kl) <kl|
#       = sum_ijkl J(ij)(kl) |ij> <kl|
#       = sum_ijkl J(ij)(kl) (|i><k| x |j><l|)
#       = sum_ijkl C(ik)(jl) (|i><k| x |j><l|)
#       = sum_jl [ sum_ik C(ik)(jl) |i><k| ] x |j><l| (using Note above)
#       = sum_jl Phi(|j><l|) x |j><l|   (using definition Phi(|i><j|) = sum_kl C[(kl)(ij)] |k><l|)
#  which is the original J(Phi) expression.
#
# This can be written equivalently as:
#  J(Phi) = sum_(0<i,j<n) Phi(Eij) otimes Eij
#  where Eij is the matrix unit with a single element in the (i,j)-th position, i.e. Eij == |i><j|

def jamiolkowski_iso(gateMx, gateMxBasis='gm', choiMxBasis='gm'):
    """
    Given a gate matrix, return the corresponding Choi matrix that is normalized
    to have trace == 1.

    Parameters
    ----------
    gateMx : numpy array
        the gate matrix to compute Choi matrix of.

    gateMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    choiMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        the Choi matrix, normalized to have trace == 1, in the desired basis.
    """
    gateMxBasis, choiMxBasis = _bt.build_basis_pair(gateMx, gateMxBasis, choiMxBasis)
    gateMx = _np.asarray(gateMx)
    gateMxInStdBasis = _bt.change_basis(gateMx, gateMxBasis, gateMxBasis.std_equivalent())

    #expand gate matrix so it acts on entire space of dmDim x dmDim density matrices
    #  so that we can take dot products with the BVec matrices below
    gateMxInStdBasis = _bt.resize_std_mx(gateMxInStdBasis, 'expand', gateMxBasis.std_equivalent(), gateMxBasis.expanded_std_equivalent())

    N = gateMxInStdBasis.shape[0] #dimension of the full-basis (expanded) gate
    dmDim = int(round(_np.sqrt(N))) #density matrix dimension

    #Note: we need to use the *full* basis of Matrix Unit, Gell-Mann, or Pauli-product matrices when
    # generating the Choi matrix, even when the original gate matrix doesn't include the entire basis.
    # This is because even when the original gate matrix doesn't include a certain basis element (B0 say),
    # conjugating with this basis element and tracing, i.e. trace(B0^dag * Gate * B0), is not necessarily zero.

    #get full list of basis matrices (in std basis) -- i.e. we use dmDim 
    BVec = _basis_matrices(choiMxBasis, dmDim)

    assert len(BVec) == N, 'Expected {}, got {}'.format(len(BVec), N)  #make sure the number of basis matrices matches the dim of the gate given

    choiMx = _np.empty( (N,N), 'complex')
    for i in range(N):
        for j in range(N):
            BiBj = _np.kron( BVec[i], _np.conjugate(BVec[j]) )
            BiBj_dag = _np.transpose(_np.conjugate(BiBj))
            choiMx[i,j] = _mt.trace( _np.dot(gateMxInStdBasis, BiBj_dag) ) \
                        / _mt.trace( _np.dot( BiBj, BiBj_dag) )

    # This construction results in a Jmx with trace == dim(H) = sqrt(gateMx.shape[0]) (dimension of density matrix)
    #  but we'd like a Jmx with trace == 1, so normalize:
    choiMx_normalized = choiMx / dmDim
    return choiMx_normalized

# GStd = sum_ij Jij (BSi x BSj^*)
def jamiolkowski_iso_inv(choiMx, choiMxBasis='gm', gateMxBasis='gm'):
    """
    Given a choi matrix, return the corresponding gate matrix.  This function
    performs the inverse of jamiolkowski_iso(...).

    Parameters
    ----------
    choiMx : numpy array
        the Choi matrix, normalized to have trace == 1, to compute gate matrix for.

    choiMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    gateMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        gate matrix in the desired basis.
    """
    gateMxBasis, choiMxBasis = _bt.build_basis_pair(choiMx, gateMxBasis, choiMxBasis)
    N = choiMx.shape[0] #dimension of full-basis (expanded) gate matrix
    dmDim = int(round(_np.sqrt(N))) #density matrix dimension

    #get full list of basis matrices (in std basis)
    BVec = _basis_matrices(choiMxBasis, dmDim)
    assert(len(BVec) == N) #make sure the number of basis matrices matches the dim of the choi matrix given

    # Invert normalization
    choiMx_unnorm = choiMx * dmDim

    gateMxInStdBasis = _np.zeros( (N,N), 'complex') #in matrix unit basis of entire density matrix
    for i in range(N):
        for j in range(N):
            BiBj = _np.kron( BVec[i], _np.conjugate(BVec[j]) )
            gateMxInStdBasis += choiMx_unnorm[i,j] * BiBj

    #project gate matrix so it acts only on the space given by the desired state space blocks
    gateMxInStdBasis = _bt.resize_std_mx(gateMxInStdBasis, 'contract', 
            gateMxBasis.expanded_std_equivalent(), gateMxBasis.std_equivalent())

    #transform gate matrix into appropriate basis
    return _bt.change_basis(gateMxInStdBasis, gateMxBasis.std_equivalent(), gateMxBasis)


def fast_jamiolkowski_iso_std(gateMx, gateMxBasis):
    """
    Given a gate matrix, return the corresponding Choi matrix in the standard
    basis that is normalized to have trace == 1.

    This routine *only* computes the case of the Choi matrix being in the
    standard (matrix unit) basis, but does so more quickly than
    :func:`jamiolkowski_iso` and so is particuarly useful when only the
    eigenvalues of the Choi matrix are needed.

    Parameters
    ----------
    gateMx : numpy array
        the gate matrix to compute Choi matrix of.

    gateMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        the Choi matrix, normalized to have trace == 1, in the std basis.
    """

    #first, get gate matrix into std basis
    gateMx = _np.asarray(gateMx)
    gateMxBasis = _bt.build_basis_for_matrix(gateMx, gateMxBasis)
    gateMxInStdBasis = _bt.change_basis(gateMx, gateMxBasis, gateMxBasis.std_equivalent())

    #Shuffle indices to go from process matrix to Jamiolkowski matrix (they vectorize differently)
    N2 = gateMxInStdBasis.shape[0]; N = int(_np.sqrt(N2))
    assert(N*N == N2) #make sure N2 is a perfect square
    Jmx = gateMxInStdBasis.reshape((N,N,N,N))
    Jmx = _np.swapaxes(Jmx,1,2).flatten()
    Jmx = Jmx.reshape((N2,N2))

    # This construction results in a Jmx with trace == dim(H) = sqrt(gateMxInPauliBasis.shape[0])
    #  but we'd like a Jmx with trace == 1, so normalize:
    Jmx_norm = Jmx / N
    return Jmx_norm

def fast_jamiolkowski_iso_std_inv(choiMx, gateMxBasis):
    """
    Given a choi matrix in the standard basis, return the corresponding gate matrix.
    This function performs the inverse of fast_jamiolkowski_iso_std(...).

    Parameters
    ----------
    choiMx : numpy array
        the Choi matrix in the standard (matrix units) basis, normalized to
        have trace == 1, to compute gate matrix for.

    gateMxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        gate matrix in the desired basis.
    """

    #Shuffle indices to go from process matrix to Jamiolkowski matrix (they vectorize differently)
    N2 = choiMx.shape[0]; N = int(_np.sqrt(N2))
    assert(N*N == N2) #make sure N2 is a perfect square
    gateMxInStdBasis = choiMx.reshape((N,N,N,N)) * N
    gateMxInStdBasis = _np.swapaxes(gateMxInStdBasis,1,2).flatten()
    gateMxInStdBasis = gateMxInStdBasis.reshape((N2,N2))
    gateMxBasis = _bt.build_basis_for_matrix(gateMxInStdBasis, gateMxBasis)

    #project gate matrix so it acts only on the space given by the desired state space blocks
    gateMxInStdBasis = _bt.resize_std_mx(gateMxInStdBasis, 'contract', 
            gateMxBasis.expanded_std_equivalent(), gateMxBasis.std_equivalent())

    #transform gate matrix into appropriate basis
    return _bt.change_basis(gateMxInStdBasis, gateMxBasis.std_equivalent(), gateMxBasis)


def sum_of_negative_choi_evals(gateset, weights=None):
    """
    Compute the amount of non-CP-ness of a gateset by summing the negative
    eigenvalues of the Choi matrix for each gate in gateset.

    Parameters
    ----------
    gateset : GateSet
        The gate set to act on.

    weights : dict
        A dictionary of weights used to multiply the negative
        eigenvalues of different gates.  Keys are gate labels, values
        are floating point numbers.

    Returns
    -------
    float
        the sum of negative eigenvalues of the Choi matrix for each gate.
    """
    if weights is not None:
        default = weights.get('gates',1.0)
        sums = sums_of_negative_choi_evals(gateset)
        return sum( [s*weights.get(gl,default) 
                     for gl,s in zip(gateset.gates.keys(),sums)] )
    else:
        return sum(sums_of_negative_choi_evals(gateset))


def sums_of_negative_choi_evals(gateset):
    """
    Compute the amount of non-CP-ness of a gateset by summing the negative
    eigenvalues of the Choi matrix for each gate in gateset separately.

    Parameters
    ----------
    gateset : GateSet
        The gate set to act on.

    Returns
    -------
    list of floats
        each element == sum of the negative eigenvalues of the Choi matrix
        for the corresponding gate (as ordered  by gateset.gates.iteritems()).
    """
    ret = []
    for (_, gate) in gateset.gates.items():
        J = fast_jamiolkowski_iso_std(gate, gateset.basis) #Choi mx basis doesn't matter
        evals = _np.linalg.eigvals( J )  #could use eigvalsh, but wary of this since eigh can be wrong...
        sumOfNeg = 0.0
        for ev in evals:
            if ev.real < 0: sumOfNeg -= ev.real
        ret.append(sumOfNeg)
    return ret


def mags_of_negative_choi_evals(gateset):
    """
    Compute the magnitudes of the negative eigenvalues of the Choi matricies
    for each gate in gateset.

    Parameters
    ----------
    gateset : GateSet
        The gate set to act on.

    Returns
    -------
    list of floats
        list of the magnitues of all negative Choi eigenvalues.  The length of
        this list will vary based on how many negative eigenvalues are found,
        as positive eigenvalues contribute nothing to this list.
    """
    ret = []
    for (_, gate) in gateset.gates.items():
        J = jamiolkowski_iso( gate, gateset.basis, choiMxBasis=gateset.basis.expanded_std_equivalent())
        evals = _np.linalg.eigvals( J )  #could use eigvalsh, but wary of this since eigh can be wrong...
        for ev in evals:
            ret.append( -ev.real if ev.real < 0 else 0.0 )
    return ret
