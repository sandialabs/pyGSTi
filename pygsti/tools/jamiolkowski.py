"""
Utility functions related to the Choi representation of gates.
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

from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.baseobjs.basis import Basis as _Basis


# Gate Mx G:      rho  --> G rho                    where G and rho are in the Pauli basis (by definition/convention)                                           # noqa
#            vec(rhoS) --> GStd vec(rhoS)           where GS and rhoS are in the std basis, GS = PtoS * G * StoP                                                # noqa
# Choi Mx J:     rho  --> sum_ij Jij Bi rho Bj^dag  where Bi is some basis of mxs for rho-space; independent of basis for rho and Bi                            # noqa
#           vec(rhoS) --> sum_ij Jij (BSi x BSj^*) vec(rhoS)  where rhoS and BSi's are in std basis                                                             # noqa
#  Now,                                                                                                                                                         # noqa
#       Jkl = Trace( sum_ij Jij (BSi x BSj^*) , (BSk x BSl^*)^dag ) / Trace( (BSk x BSl^*), (BSk x BSl^*)^dag )                                                 # noqa
#           = Trace( GStd , (BSk x BSl^*)^dag ) / Trace( (BSk x BSl^*), (BSk x BSl^*)^dag )                                                                     # noqa
#  In below function, take Bi's to be Pauli matrices
#  Note: vec(.) vectorization above is assumed to be done by-*rows* (as numpy.flatten does).

# Note that in just the std basis, the construction of the Jamiolkowski representation of a process phi is                                                      # noqa
#  J(Phi) = sum_(0<i,j<n) Phi(|i><j|) x |i><j|    where {|i>}_1^n spans the state space                                                                         # noqa
#
#  Derivation: if we write:                                                                                                                                     # noqa
#    Phi(|i><j|) = sum_kl C[(kl)(ij)] |k><l|                                                                                                                    # noqa
#  and                                                                                                                                                          # noqa
#    rho = sum_ij rho_ij |i><j|                                                                                                                                 # noqa
#  then                                                                                                                                                         # noqa
#    Phi(rho) = sum_(ij)(kl) C[(kl)(ij)] rho_ij |k><l|                                                                                                          # noqa
#             = sum_(ij)(kl) C[(kl)(ij)] |k> rho_ij <l|                                                                                                         # noqa
#             = sum_(ij)(kl) C[(kl)(ij)] |k> <i| rho |j> <l|                                                                                                    # noqa
#             = sum_(ij)(kl) C[(ik)(jl)] |i> <j| rho |l> <k|  (just permute index labels)                                                                       # noqa
#  The definition of the Jamiolkoski matrix J is:                                                                                                               # noqa
#    Phi(rho) = sum_(ij)(kl) J(ij)(kl) |i><j| rho |l><k|                                                                                                        # noqa
#  so                                                                                                                                                           # noqa
#    J(ij)(kl) == C[(ik)(jl)]                                                                                                                                   # noqa
#
#  Note: |i><j| x |k><l| is an object in "gate/process" space, since                                                                                            # noqa
#    it maps a vectorized density matrix, e.g. |a><b| to another density matrix via:                                                                            # noqa
#    (|i><j| x |k><l|) vec(|a><b|) = [mx with 1 in (i*dmDim + k) row and (j*dmDim + l) col][vec with 1 in a*dmDim+b row]                                        # noqa
#                                  = vec(|i><k|) if |a><b| == |j><l| else 0                                                                                     # noqa
#    so (|i><j| x |k><l|) vec(|j><l|) = vec(|i><k|)                                                                                                             # noqa
#    and could write as: (|ik><jl|) |jl> = |ik>                                                                                                                 # noqa
#
# Now write J as:                                                                                                                                               # noqa
#    J  = sum_ijkl |ij> J(ij)(kl) <kl|                                                                                                                          # noqa
#       = sum_ijkl J(ij)(kl) |ij> <kl|                                                                                                                          # noqa
#       = sum_ijkl J(ij)(kl) (|i><k| x |j><l|)                                                                                                                  # noqa
#       = sum_ijkl C(ik)(jl) (|i><k| x |j><l|)                                                                                                                  # noqa
#       = sum_jl [ sum_ik C(ik)(jl) |i><k| ] x |j><l| (using Note above)                                                                                        # noqa
#       = sum_jl Phi(|j><l|) x |j><l|   (using definition Phi(|i><j|) = sum_kl C[(kl)(ij)] |k><l|)                                                              # noqa
#  which is the original J(Phi) expression.                                                                                                                     # noqa
#
# This can be written equivalently as:                                                                                                                          # noqa
#  J(Phi) = sum_(0<i,j<n) Phi(Eij) otimes Eij                                                                                                                   # noqa
#  where Eij is the matrix unit with a single element in the (i,j)-th position, i.e. Eij == |i><j|                                                              # noqa

def jamiolkowski_iso(operation_mx, op_mx_basis='pp', choi_mx_basis='pp'):
    """
    Given a operation matrix, return the corresponding Choi matrix that is normalized to have trace == 1.

    Parameters
    ----------
    operation_mx : numpy array
        the operation matrix to compute Choi matrix of.

    op_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    choi_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        the Choi matrix, normalized to have trace == 1, in the desired basis.
    """
    operation_mx = _np.asarray(operation_mx)
    op_mx_basis = _bt.create_basis_for_matrix(operation_mx, op_mx_basis)
    opMxInStdBasis = _bt.change_basis(operation_mx, op_mx_basis, op_mx_basis.create_equivalent('std'))

    #expand operation matrix so it acts on entire space of dmDim x dmDim density matrices
    #  so that we can take dot products with the BVec matrices below
    opMxInStdBasis = _bt.resize_std_mx(opMxInStdBasis, 'expand', op_mx_basis.create_equivalent(
        'std'), op_mx_basis.create_simple_equivalent('std'))

    N = opMxInStdBasis.shape[0]  # dimension of the full-basis (expanded) gate
    dmDim = int(round(_np.sqrt(N)))  # density matrix dimension

    #Note: we need to use the *full* basis of Matrix Unit, Gell-Mann, or Pauli-product matrices when
    # generating the Choi matrix, even when the original operation matrix doesn't include the entire basis.
    # This is because even when the original operation matrix doesn't include a certain basis element (B0 say),
    # conjugating with this basis element and tracing, i.e. trace(B0^dag * Operation * B0), is not necessarily zero.

    #get full list of basis matrices (in std basis) -- i.e. we use dmDim
    if not isinstance(choi_mx_basis, _Basis):
        choi_mx_basis = _Basis.cast(choi_mx_basis, N)  # we'd like a basis of dimension N

    BVec = choi_mx_basis.create_simple_equivalent().elements
    M = len(BVec)  # can be < N if basis has multiple block dims
    assert(M == N), 'Expected {}, got {}'.format(M, N)

    choiMx = _np.empty((N, N), 'complex')
    for i in range(M):
        for j in range(M):
            BiBj = _np.kron(BVec[i], _np.conjugate(BVec[j]))
            num = _np.vdot(BiBj, opMxInStdBasis)
            den = _np.linalg.norm(BiBj) ** 2
            choiMx[i, j] = num / den 

    # This construction results in a Jmx with trace == dim(H) = sqrt(operation_mx.shape[0])
    #  (dimension of density matrix) but we'd like a Jmx with trace == 1, so normalize:
    choiMx_normalized = choiMx / dmDim
    return choiMx_normalized

# GStd = sum_ij Jij (BSi x BSj^*)


def jamiolkowski_iso_inv(choi_mx, choi_mx_basis='pp', op_mx_basis='pp'):
    """
    Given a choi matrix, return the corresponding operation matrix.

    This function performs the inverse of :func:`jamiolkowski_iso`.

    Parameters
    ----------
    choi_mx : numpy array
        the Choi matrix, normalized to have trace == 1, to compute operation matrix for.

    choi_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    op_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        operation matrix in the desired basis.
    """
    choi_mx = _np.asarray(choi_mx)  # will have "expanded" dimension even if bases are for reduced...
    N = choi_mx.shape[0]  # dimension of full-basis (expanded) operation matrix
    if not isinstance(choi_mx_basis, _Basis):  # if we're not given a basis, build
        choi_mx_basis = _Basis.cast(choi_mx_basis, N)  # one with the full dimension

    dmDim = int(round(_np.sqrt(N)))  # density matrix dimension

    #get full list of basis matrices (in std basis)
    BVec = _bt.basis_matrices(choi_mx_basis.create_simple_equivalent(), N)
    assert(len(BVec) == N)  # make sure the number of basis matrices matches the dim of the choi matrix given

    # Invert normalization
    choiMx_unnorm = choi_mx * dmDim

    opMxInStdBasis = _np.zeros((N, N), 'complex')  # in matrix unit basis of entire density matrix
    for i in range(N):
        for j in range(N):
            BiBj = _np.kron(BVec[i], _np.conjugate(BVec[j]))
            opMxInStdBasis += choiMx_unnorm[i, j] * BiBj

    if not isinstance(op_mx_basis, _Basis):
        op_mx_basis = _Basis.cast(op_mx_basis, N)  # make sure op_mx_basis is a Basis; we'd like dimension to be N

    #project operation matrix so it acts only on the space given by the desired state space blocks
    opMxInStdBasis = _bt.resize_std_mx(opMxInStdBasis, 'contract',
                                       op_mx_basis.create_simple_equivalent('std'),
                                       op_mx_basis.create_equivalent('std'))

    #transform operation matrix into appropriate basis
    return _bt.change_basis(opMxInStdBasis, op_mx_basis.create_equivalent('std'), op_mx_basis)


def fast_jamiolkowski_iso_std(operation_mx, op_mx_basis):
    """
    The corresponding Choi matrix in the standard basis that is normalized to have trace == 1.

    This routine *only* computes the case of the Choi matrix being in the
    standard (matrix unit) basis, but does so more quickly than
    :func:`jamiolkowski_iso` and so is particuarly useful when only the
    eigenvalues of the Choi matrix are needed.

    Parameters
    ----------
    operation_mx : numpy array
        the operation matrix to compute Choi matrix of.

    op_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        the Choi matrix, normalized to have trace == 1, in the std basis.
    """

    #first, get operation matrix into std basis
    operation_mx = _np.asarray(operation_mx)
    op_mx_basis = _bt.create_basis_for_matrix(operation_mx, op_mx_basis)
    opMxInStdBasis = _bt.change_basis(operation_mx, op_mx_basis, op_mx_basis.create_equivalent('std'))

    #expand operation matrix so it acts on entire space of dmDim x dmDim density matrices
    opMxInStdBasis = _bt.resize_std_mx(opMxInStdBasis, 'expand', op_mx_basis.create_equivalent('std'),
                                       op_mx_basis.create_simple_equivalent('std'))

    #Shuffle indices to go from process matrix to Jamiolkowski matrix (they vectorize differently)
    N2 = opMxInStdBasis.shape[0]; N = int(_np.sqrt(N2))
    assert(N * N == N2)  # make sure N2 is a perfect square
    Jmx = opMxInStdBasis.reshape((N, N, N, N))
    Jmx = _np.swapaxes(Jmx, 1, 2).ravel()
    Jmx = Jmx.reshape((N2, N2))

    # This construction results in a Jmx with trace == dim(H) = sqrt(gateMxInPauliBasis.shape[0])
    #  but we'd like a Jmx with trace == 1, so normalize:
    Jmx_norm = Jmx / N
    return Jmx_norm


def fast_jamiolkowski_iso_std_inv(choi_mx, op_mx_basis):
    """
    Given a choi matrix in the standard basis, return the corresponding operation matrix.

    This function performs the inverse of :func:`fast_jamiolkowski_iso_std`.

    Parameters
    ----------
    choi_mx : numpy array
        the Choi matrix in the standard (matrix units) basis, normalized to
        have trace == 1, to compute operation matrix for.

    op_mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        operation matrix in the desired basis.
    """

    #Shuffle indices to go from process matrix to Jamiolkowski matrix (they vectorize differently)
    N2 = choi_mx.shape[0]; N = int(_np.sqrt(N2))
    assert(N * N == N2)  # make sure N2 is a perfect square
    opMxInStdBasis = choi_mx.reshape((N, N, N, N)) * N
    opMxInStdBasis = _np.swapaxes(opMxInStdBasis, 1, 2).ravel()
    opMxInStdBasis = opMxInStdBasis.reshape((N2, N2))
    op_mx_basis = _bt.create_basis_for_matrix(opMxInStdBasis, op_mx_basis)

    #project operation matrix so it acts only on the space given by the desired state space blocks
    opMxInStdBasis = _bt.resize_std_mx(opMxInStdBasis, 'contract',
                                       op_mx_basis.create_simple_equivalent('std'),
                                       op_mx_basis.create_equivalent('std'))

    #transform operation matrix into appropriate basis
    return _bt.change_basis(opMxInStdBasis, op_mx_basis.create_equivalent('std'), op_mx_basis)

def sum_of_negative_choi_eigenvalues_gate(op_mx, op_mx_basis):
    """
    Compute the sum of the negative Choi eigenvalues of a process matrix.

    Parameters
    ----------
    op_mx : np.array

    op_mx_basis : Basis

    Returns
    -------
    float
        the sum of the negative eigenvalues of the Choi representation of op_mx
    """
    sumOfNeg = 0
    J = fast_jamiolkowski_iso_std(op_mx, op_mx_basis)  # Choi mx basis doesn't matter
    evals = _np.linalg.eigvals(J)  # could use eigvalsh, but wary of this since eigh can be wrong...
    for ev in evals:
            if ev.real < 0: sumOfNeg -= ev.real
    return sumOfNeg

def sum_of_negative_choi_eigenvalues(model, weights=None):
    """
    Compute the amount of non-CP-ness of a model.

    This is defined (somewhat arbitarily) by summing the negative
    eigenvalues of the Choi matrix for each gate in `model`.

    Parameters
    ----------
    model : Model
        The model to act on.

    weights : dict
        A dictionary of weights used to multiply the negative
        eigenvalues of different gates.  Keys are operation labels, values
        are floating point numbers.

    Returns
    -------
    float
        the sum of negative eigenvalues of the Choi matrix for each gate.
    """
    if weights is not None:
        default = weights.get('gates', 1.0)
        sums = sums_of_negative_choi_eigenvalues(model)
        return sum([s * weights.get(gl, default)
                    for gl, s in zip(model.operations.keys(), sums)])
    else:
        return sum(sums_of_negative_choi_eigenvalues(model))


def sums_of_negative_choi_eigenvalues(model):
    """
    Compute the amount of non-CP-ness of a model.

    This is defined (somewhat arbitarily) by summing the negative
    eigenvalues of the Choi matrix for each gate in model separately.
    This function is different from :func:`sum_of_negative_choi_eigenvalues`
    in that it returns sums separately for each operation of `model`.

    Parameters
    ----------
    model : Model
        The model to act on.

    Returns
    -------
    list of floats
        each element == sum of the negative eigenvalues of the Choi matrix
        for the corresponding gate (as ordered  by model.operations.iteritems()).
    """
    ret = []
    for (_, gate) in model.operations.items():
        J = fast_jamiolkowski_iso_std(gate.to_dense(), model.basis)  # Choi mx basis doesn't matter
        evals = _np.linalg.eigvals(J)  # could use eigvalsh, but wary of this since eigh can be wrong...
        sumOfNeg = 0.0
        for ev in evals:
            if ev.real < 0: sumOfNeg -= ev.real
        ret.append(sumOfNeg)
    return ret


def magnitudes_of_negative_choi_eigenvalues(model):
    """
    Compute the magnitudes of the negative eigenvalues of the Choi matricies for each gate in `model`.

    Parameters
    ----------
    model : Model
        The model to act on.

    Returns
    -------
    list of floats
        list of the magnitues of all negative Choi eigenvalues.  The length of
        this list will vary based on how many negative eigenvalues are found,
        as positive eigenvalues contribute nothing to this list.
    """
    ret = []
    for (_, gate) in model.operations.items():
        J = jamiolkowski_iso(gate, model.basis, choi_mx_basis=model.basis.create_simple_equivalent('std'))
        evals = _np.linalg.eigvals(J)  # could use eigvalsh, but wary of this since eigh can be wrong...
        for ev in evals:
            ret.append(-ev.real if ev.real < 0 else 0.0)
    return ret
