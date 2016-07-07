from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Utility functions related to the Choi representation of gates."""

import numpy as _np
from . import basistools as _bt
from . import matrixtools as _mt

# Gate Mx G:      rho  --> G rho                    where G and rho are in the Pauli basis (by definition/convention)
#            vec(rhoS) --> GStd vec(rhoS)           where GS and rhoS are in the std basis, GS = PtoS * G * StoP
# Choi Mx J:     rho  --> sum_ij Jij Bi rho Bj^dag  where Bi is some basis of mxs for rho-space; independent of basis for rho and Bi
#           vec(rhoS) --> sum_ij Jij (BSj^* x BSi) vec(rhoS)  where rhoS and BSi's are in std basis
#  Now,
#       Jkl = Trace( sum_ij Jij (BSj^* x BSi) , (BSl^* x BSk)^dag ) / Trace( (BSl^* x BSk), (BSl^* x BSk)^dag )
#           = Trace( GStd , (BSl^* x BSk)^dag ) / Trace( (BSl^* x BSk), (BSl^* x BSk)^dag )
#  In below function, take Bi's to be Pauli matrices

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



def jamiolkowski_iso(gateMx, gateMxBasis="gm", choiMxBasis="gm", dimOrStateSpaceDims=None):
    """
    Given a gate matrix, return the corresponding Choi matrix that is normalized
    to have trace == 1.

    Parameters
    ----------
    gateMx : numpy array
        the gate matrix to compute Choi matrix of.

    gateMxBasis : {"std","gm","pp"}, optional
        the basis of gateMx: standard (matrix units), Gell-Mann, or Pauli-product,
        respectively.

    choiMxBasis : {"std","gm","pp"}, optional
        the basis for the returned Choi matrix: standard (matrix units), Gell-Mann,
        or Pauli-product, respectively.

    dimOrStateSpaceDims : int or list of ints, optional
        Structure of the density-matrix space, which further specifies the basis
        of gateMx (see BasisTools).

    Returns
    -------
    numpy array
        the Choi matrix, normalized to have trace == 1, in the desired basis.
    """

    #first, get gate matrix into std basis
    gateMx = _np.asarray(gateMx)
    if gateMxBasis == "std":
        gateMxInStdBasis = gateMx
    elif gateMxBasis == "gm" or gateMxBasis == "pauli":
        gateMxInStdBasis = _bt.gm_to_std(gateMx, dimOrStateSpaceDims)
    elif gateMxBasis == "pp":
        gateMxInStdBasis = _bt.pp_to_std(gateMx, dimOrStateSpaceDims)
    else: raise ValueError("Invalid gateMxBasis: %s" % gateMxBasis)

    #expand gate matrix so it acts on entire space of dmDim x dmDim density matrices
    #  so that we can take dot products with the BVec matrices below
    gateMxInStdBasis = _bt.expand_from_std_direct_sum_mx(gateMxInStdBasis, dimOrStateSpaceDims)
    N = gateMxInStdBasis.shape[0] #dimension of the full-basis (expanded) gate
    dmDim = int(round(_np.sqrt(N))) #density matrix dimension

    #Note: we need to use the *full* basis of Matrix Unit, Gell-Mann, or Pauli-product matrices when
    # generating the Choi matrix, even when the original gate matrix doesn't include the entire basis.
    # This is because even when the original gate matrix doesn't include a certain basis element (B0 say),
    # conjugating with this basis element and tracing, i.e. trace(B0^dag * Gate * B0), is not necessarily zero.

    #get full list of basis matrices (in std basis) -- i.e. we use dmDim not dimOrStateSpaceDims
    if choiMxBasis == "gm":
        BVec = _bt.gm_matrices(dmDim)
    elif choiMxBasis == "std":
        BVec = _bt.std_matrices(dmDim)
    elif choiMxBasis == "pp":
        BVec = _bt.pp_matrices(dmDim)
    else: raise ValueError("Invalid choiMxBasis: %s" % choiMxBasis)
    assert(len(BVec) == N) #make sure the number of basis matrices matches the dim of the gate given

    choiMx = _np.empty( (N,N), 'complex')
    for i in range(N):
        for j in range(N):
            BjBi = _np.kron( _np.conjugate(BVec[j]), BVec[i] )
            BjBi_dag = _np.transpose(_np.conjugate(BjBi))
            choiMx[i,j] = _mt.trace( _np.dot(gateMxInStdBasis, BjBi_dag) ) \
                        / _mt.trace( _np.dot( BjBi, BjBi_dag) )

    # This construction results in a Jmx with trace == dim(H) = sqrt(gateMx.shape[0]) (dimension of density matrix)
    #  but we'd like a Jmx with trace == 1, so normalize:
    choiMx_normalized = choiMx / dmDim
    return choiMx_normalized

# GStd = sum_ij Jij (BSj^* x BSi)
def jamiolkowski_iso_inv(choiMx, choiMxBasis="gm", gateMxBasis="gm", dimOrStateSpaceDims=None):
    """
    Given a choi matrix, return the corresponding gate matrix.  This function
    performs the inverse of jamiolkowski_iso(...).

    Parameters
    ----------
    choiMx : numpy array
        the Choi matrix, normalized to have trace == 1, to compute gate matrix for.

    choiMxBasis : {"std","gm","pp"}, optional
        the basis of choiMx: standard (matrix units), Gell-Mann, or Pauli-product,
        respectively.

    gateMxBasis : {"std","gm","pp"}, optional
        the basis for the returned gate matrix: standard (matrix units), Gell-Mann,
        or Pauli-product, respectively.

    dimOrStateSpaceDims : int or list of ints, optional
        Structure of the density-matrix space, which further specifies the basis
        of the returned gateMx (see BasisTools).

    Returns
    -------
    numpy array
        gate matrix in the desired basis.
    """
    N = choiMx.shape[0] #dimension of full-basis (expanded) gate matrix
    dmDim = int(round(_np.sqrt(N))) #density matrix dimension

    #get full list of basis matrices (in std basis)
    if choiMxBasis == "gm":
        BVec = _bt.gm_matrices(dmDim)
    elif choiMxBasis == "std":
        BVec = _bt.std_matrices(dmDim)
    elif choiMxBasis == "pp":
        BVec = _bt.pp_matrices(dmDim)
    else: raise ValueError("Invalid choiMxBasis: %s" % choiMxBasis)
    assert(len(BVec) == N) #make sure the number of basis matrices matches the dim of the choi matrix given

    # Invert normalization
    choiMx_unnorm = choiMx * dmDim

    gateMxInStdBasis = _np.zeros( (N,N), 'complex') #in matrix unit basis of entire density matrix
    for i in range(N):
        for j in range(N):
            BjBi = _np.kron( _np.conjugate(BVec[j]), BVec[i] )
            gateMxInStdBasis += choiMx_unnorm[i,j] * BjBi

    #project gate matrix so it acts only on the space given by the desired state space blocks
    gateMxInStdBasis = _bt.contract_to_std_direct_sum_mx(gateMxInStdBasis, dimOrStateSpaceDims)

    #transform gate matrix into appropriate basis
    bReal = False
    if gateMxBasis == "std":
        gateMx = gateMxInStdBasis
    elif gateMxBasis == "gm" or gateMxBasis == "pauli":
        gateMx = _bt.std_to_gm(gateMxInStdBasis, dimOrStateSpaceDims); bReal = True
    elif gateMxBasis == "pp":
        gateMx = _bt.std_to_pp(gateMxInStdBasis, dimOrStateSpaceDims); bReal = True
    else: raise ValueError("Invalid gateMxBasis: %s" % gateMxBasis)

    if bReal: # matrix should always be real
        assert( _np.max(abs(_np.imag(gateMx))) < 1e-8 )
        gateMx = _np.real(gateMx)

    return gateMx



## Gate Mx G:      rho  --> G rho                    where G and rho are in the Pauli basis (by definition/convention)
##            vec(rhoS) --> GStd vec(rhoS)           where GS and rhoS are in the std basis, GS = PtoS * G * StoP
## Choi Mx J:     rho  --> sum_ij Jij Bi rho Bj^dag  where Bi is some basis of mxs for rho-space; independent of basis for rho and Bi
##           vec(rhoS) --> sum_ij Jij (BSj^* x BSi) vec(rhoS)  where rhoS and BSi's are in std basis
##  Now,
##       Jkl = Trace( sum_ij Jij (BSj^* x BSi) , (BSl^* x BSk)^dag ) / Trace( (BSl^* x BSk), (BSl^* x BSk)^dag )
##           = Trace( GStd , (BSl^* x BSk)^dag ) / Trace( (BSl^* x BSk), (BSl^* x BSk)^dag )
##  In below function, take Bi's to be Pauli matrices
#def _opWithJamiolkowskiIsomorphism_old(gateMxInPauliBasis, Jbasis):
#    """
#    Given a gate matrix in the Pauli basis, return the corresponding
#      Choi matrix in the basis given by Jbasis.  The allowed values
#      of Jbasis are "MatrixUnit" and "Pauli".  The returned Choi
#      matrix is normalized to have trace == 1.
#    """
#    N = gateMxInPauliBasis.shape[0]
#    if N == 4:  # 1-qubit case
#        ConvToStd = PauliToStd; ConvFromStd = StdToPauli #always Pauli conversion since gateMxInPauliBasis is
#        if Jbasis == "MatrixUnit": BVec = _bt.mxUnitVec  #vector of basis elements, each in std basis
#        elif Jbasis == "Pauli":    BVec = _bt.sigmaVec   #vector of basis elements, each in std basis
#        else: raise ValueError("jamiolkowski_iso: %s basis is not implemented." % Jbasis)
#    elif N == 16: # 2-qubit case
#        ConvToStd = PauliToStd_2Q; ConvFromStd = StdToPauli_2Q  #always Pauli conversion since gateMxInPauliBasis is
#        if Jbasis == "MatrixUnit": BVec = _bt.mxUnitVec_2Q  #vector of basis elements, each in std basis
#        elif Jbasis == "Pauli":    BVec = _bt.sigmaVec_2Q   #vector of basis elements, each in std basis
#        else: raise ValueError("jamiolkowski_iso: %s basis is not implemented." % Jbasis)
#    else:
#        raise ValueError("jamiolkowski_iso: only one and two qubit cases are currently implemented.")
#
#    gateMxInStdBasis = _np.zeros( (N,N), 'complex')
#    for i in range(N):
#        for j in range(N):
#            BjBi = _np.kron( _np.conjugate(BVec[j]), BVec[i] )
#            gateMxInStdBasis += Jmx_unnorm[i,j] * BjBi
#
#    gateMxInPauliBasis = _np.dot(ConvFromStd, _np.dot(gateMxInStdBasis, ConvToStd) )
#    assert( _np.max(abs(_np.imag(gateMxInPauliBasis))) < 1e-8 ) # should always be real
#    return _np.real(gateMxInPauliBasis)
#    #return gateMxInPauliBasis  #for debugging
#
#
#
#def _opWithJamiolkowskiIsomorphism_mxunit(gateMxInPauliBasis):
#    if gateMxInPauliBasis.shape[0] == 4:  # 1-qubit case
#        gateMxInStdBasis = _np.dot( PauliToStd, _np.dot(gateMxInPauliBasis, StdToPauli) )  #process mx for gate
#    elif gateMxInPauliBasis.shape[0] == 16: # 2-qubit case
#        gateMxInStdBasis = _np.dot( PauliToStd_2Q, _np.dot(gateMxInPauliBasis, StdToPauli_2Q) )  #process mx for gate
#    else:
#        raise ValueError("jamiolkowski_iso: only one and two qubit cases are currently implemented.")
#
#    def iVM(i,N): # vectorized index i to (row, col) index of matrix
#        col = i // N
#        row = i - N*col
#        return ( row, col )
#
#    def iMV(row, col, N): # (row, col) of matrix to vectorized index
#        return col * N + row
#
#    #Shuffle indices to go from process matrix to Jamiolkowski matrix (they vectorize differently)
#    N2 = gateMxInStdBasis.shape[0]; N = _np.sqrt(N2)
#    Jmx = _np.empty( (N2,N2), 'complex' )
#    for i in range(N2):
#        (irow, icol) = iVM(i,N)
#        for j in range(N2):
#            (jrow,jcol) = iVM(j,N)
#            k = iMV(irow, jrow, N)
#            l = iMV(icol, jcol, N)
#            Jmx[k,l] = gateMxInStdBasis[i,j]
#
#    # This construction results in a Jmx with trace == dim(H) = sqrt(gateMxInPauliBasis.shape[0])
#    #  but we'd like a Jmx with trace == 1, so normalize:
#    Jmx_norm = Jmx / _np.sqrt(gateMxInPauliBasis.shape[0])
#    return Jmx_norm
#
#def _opWithInvJamiolkowskiIsomorphism_mxunit(Jmx_norm):
#
#    def iVM(i,N): # vectorized index i to (row, col) index of matrix
#        col = i // N
#        row = i - N*col
#        return ( row, col )
#
#    def iMV(row, col, N): # (row, col) of matrix to vectorized index
#        return col * N + row
#
#    # Invert normalization
#    Jmx = Jmx_norm * _np.sqrt(Jmx_norm.shape[0])
#
#    #Shuffle indices to go from Jamiolkowski matrix to process matrix (they vectorize differently)
#    N2 = Jmx.shape[0]; N = _np.sqrt(N2)
#    gateMxInStdBasis = _np.empty( (N2,N2), 'complex' )
#    for i in range(N2):
#        (irow, icol) = iVM(i,N)
#        for j in range(N2):
#            (jrow,jcol) = iVM(j,N)
#            k = iMV(irow, jrow, N)
#            l = iMV(icol, jcol, N)
#            gateMxInStdBasis[i,j] = Jmx[k,l]
#
#    if gateMxInStdBasis.shape[0] == 4:  # 1-qubit case
#        gateMxInPauliBasis = _np.dot(StdToPauli, _np.dot(gateMxInStdBasis, PauliToStd) )
#    elif gateMxInStdBasis.shape[0] == 16: # 2-qubit case
#        gateMxInPauliBasis = _np.dot(StdToPauli_2Q, _np.dot(gateMxInStdBasis, PauliToStd_2Q) )
#    else:
#        raise ValueError("jamiolkowski_iso: only one and two qubit cases are currently implemented.")
#    #return gateMxInPauliBasis #for debugging -- should always be real
#    return _np.real(gateMxInPauliBasis)



def sum_of_negative_choi_evals(gateset):
    """
    Compute the amount of non-CP-ness of a gateset by summing the negative
    eigenvalues of the Choi matrix for each gate in gateset.

    Parameters
    ----------
    gateset : GateSet
        The gate set to act on.

    Returns
    -------
    float
        the sum of negative eigenvalues of the Choi matrix for each gate.
    """
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
        J = jamiolkowski_iso( gate, choiMxBasis="std" )
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
        J = jamiolkowski_iso( gate, choiMxBasis="std" )
        evals = _np.linalg.eigvals( J )  #could use eigvalsh, but wary of this since eigh can be wrong...
        for ev in evals:
            ret.append( -ev.real if ev.real < 0 else 0.0 )
    return ret
