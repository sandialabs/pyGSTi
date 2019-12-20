""" Routines for building qutrit gates and models """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy import linalg as _linalg

from .. import objects as _objs
from ..tools import unitary_to_process_mx, change_basis, Basis


#Define 2 qubit to symmetric (+) antisymmetric space transformation A:
A = _np.matrix([[1, 0, 0, 0],
                #               [0,0,0,1],
                [0, 1. / _np.sqrt(2), 1. / _np.sqrt(2), 0],
                [0, 1. / _np.sqrt(2), -1. / _np.sqrt(2), 0],
                [0, 0, 0, 1], ])

X = _np.matrix([[0, 1], [1, 0]])
Y = _np.matrix([[0, -1j], [1j, 0]])


def X2qubit(theta):
    """ Returns X(theta)^\otimes 2 (2-qubit 'XX' unitary)"""
    x = _np.matrix(_linalg.expm(-1j / 2. * theta * _np.matrix([[0, 1], [1, 0]])))
    return _np.kron(x, x)


def Y2qubit(theta):
    """ Returns Y(theta)^\otimes 2 (2-qubit 'YY' unitary)"""
    y = _np.matrix(_linalg.expm(-1j / 2. * theta * _np.matrix([[0, -1j], [1j, 0]])))
    return _np.kron(y, y)


def ms2qubit(theta, phi):
    """ Returns Molmer-Sorensen gate for two qubits """
    return _np.matrix(_linalg.expm(-1j / 2 * theta
                                   * _np.kron(
                                       _np.cos(phi) * X + _np.sin(phi) * Y,
                                       _np.cos(phi) * X + _np.sin(phi) * Y)
                                   ))

#Projecting above gates into symmetric subspace (qutrit space)
#(state space ordering is |0> = |00>, |1> ~ |01>+|10>,|2>=|11>, so state |i> corresponds to i detector counts


#Removes columns and rows from inputArr
def _remove_from_matrix(inputArr, columns, rows, outputType=_np.matrix):
    inputArr = _np.array(inputArr)
    return outputType([
        [inputArr[row_num][col_num]
            for col_num in range(len(inputArr[row_num]))
            if col_num not in columns]

        for row_num in range(len(inputArr))
        if row_num not in rows])


def to_qutrit_space(inputMat):
    """ Projects a 2-qubit unitary matrix onto the symmetric "qutrit space" """
    inputMat = _np.matrix(inputMat)
    return _remove_from_matrix(A * inputMat * A**-1, [2], [2])
#    return (A * inputMat * A**-1)[:3,:3]#Comment out above line and uncomment this line if you want the state space
#labelling to be |0>=|00>,|1>=|11>,|2>~|01>+|10>


def MS3(theta, phi):
    """ Returns Qutrit Molmer-Sorenson unitary """
    return to_qutrit_space(ms2qubit(theta, phi))


def XX3(theta):
    """ Returns Qutrit XX unitary """
    return to_qutrit_space(X2qubit(theta))


def YY3(theta):
    """ Returns Qutrit YY unitary """
    return to_qutrit_space(Y2qubit(theta))


def _random_rot(scale, arrType=_np.array, seed=None):
    rndm = _np.random.RandomState(seed)
    randH = scale * (rndm.randn(3, 3) + 1j * rndm.randn(3, 3))
    randH = _np.dot(_np.conj(randH.T), randH)
    randU = _linalg.expm(-1j * randH)
    return arrType(randU)


def make_qutrit_model(errorScale, Xangle=_np.pi / 2, Yangle=_np.pi / 2,
                      MSglobal=_np.pi / 2, MSlocal=0,
                      similarity=False, seed=None, basis='qt'):
    """
    Constructs a standard qutrit :class:`Model` containing the identity,
    XX, YY, and Molmer-Sorenson gates.

    Parameters
    ----------
    errorScale : float
        Magnitude of random rotations to apply to the returned model.  If
        zero, then perfect "ideal" gates are constructed.

    Xangle, Yangle : float
        The angle of the single-qubit 'X' and 'Y' rotations in the 'XX' and 'YY'
        gates.  An X-rotation by `theta` is given by `U = exp(-i/2 * theta * X)`
        where `X` is a Pauli matrix, and likewise for the Y-rotation.

    MSglobal, MSlocal : float
        "Global" and "local" angles for the Molmer-Sorenson gate, defined by
        the corresponding 2-qubit unitary:

        `U = exp(-i/2 * MSglobal * (cos(MSlocal)*X + sin(MSlocal)*Y)^2)`

        where `x^2` means the *tensor product* of `x` with itself.

    similarity : bool, optional
        If true, then apply the random rotations (whose strengths are given
        by `errorScale`) as similarity transformations rather than just as
        post-multiplications to the ideal operation matrices.

    seed : int, optional
        The seed used to generate random rotations.

    basis : str, optional
        The string abbreviation of the basis of the returned vector.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm) and Qutrit (qt).  A `Basis`
        object may also be used.

    Returns
    -------
    Model
    """
    arrType = _np.array  # Are we casting gates as matrices or arrays?

    rho0 = arrType(([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]))

    identity3 = arrType(_np.identity(3))

    E0 = arrType(_np.diag([1, 0, 0]))
    E1 = arrType(_np.diag([0, 1, 0]))
    E2 = arrType(_np.diag([0, 0, 1]))

    #Define gates as unitary ops on Hilbert space
    gateImx = arrType(identity3)
    gateXmx = arrType(XX3(Xangle))
    gateYmx = arrType(YY3(Yangle))
    gateMmx = arrType(MS3(MSglobal, MSlocal))

    #Now introduce unitary noise.

    scale = errorScale
    Xrand = _random_rot(scale, seed=seed)
    Yrand = _random_rot(scale)
    Mrand = _random_rot(scale)
    Irand = _random_rot(scale)

    if similarity:  # Change basis for each gate; this preserves rotation angles, and should map identity to identity
        gateXmx = _np.dot(_np.dot(_np.conj(Xrand).T, gateXmx), Xrand)
        gateYmx = _np.dot(_np.dot(_np.conj(Yrand).T, gateYmx), Yrand)
        gateMmx = _np.dot(_np.dot(_np.conj(Mrand).T, gateMmx), Mrand)
        gateImx = _np.dot(_np.dot(_np.conj(Irand).T, gateMmx), Irand)

    else:
        gateXmx = _np.dot(gateXmx, Xrand)
        gateYmx = _np.dot(gateYmx, Yrand)
        gateMmx = _np.dot(gateMmx, Mrand)
        gateImx = _np.dot(gateImx, Mrand)

    #Change gate representation to superoperator in Gell-Mann basis
    gateISO = unitary_to_process_mx(gateImx)
    gateISOfinal = change_basis(gateISO, "std", basis)
    gateXSO = unitary_to_process_mx(gateXmx)
    gateXSOfinal = change_basis(gateXSO, "std", basis)
    gateYSO = unitary_to_process_mx(gateYmx)
    gateYSOfinal = change_basis(gateYSO, "std", basis)
    gateMSO = unitary_to_process_mx(gateMmx)
    gateMSOfinal = change_basis(gateMSO, "std", basis)

    rho0final = change_basis(_np.reshape(rho0, (9, 1)), "std", basis)
    E0final = change_basis(_np.reshape(E0, (9, 1)), "std", basis)
    E1final = change_basis(_np.reshape(E1, (9, 1)), "std", basis)
    E2final = change_basis(_np.reshape(E2, (9, 1)), "std", basis)

    sslbls = _objs.StateSpaceLabels(['QT'], [9])
    qutritMDL = _objs.ExplicitOpModel(sslbls, Basis.cast(basis, 9))
    qutritMDL.preps['rho0'] = rho0final
    qutritMDL.povms['Mdefault'] = _objs.UnconstrainedPOVM([('0bright', E0final),
                                                           ('1bright', E1final),
                                                           ('2bright', E2final)])
    qutritMDL.operations['Gi'] = _objs.FullDenseOp(arrType(gateISOfinal))
    qutritMDL.operations['Gx'] = _objs.FullDenseOp(arrType(gateXSOfinal))
    qutritMDL.operations['Gy'] = _objs.FullDenseOp(arrType(gateYSOfinal))
    qutritMDL.operations['Gm'] = _objs.FullDenseOp(arrType(gateMSOfinal))
    qutritMDL.default_gauge_group = _objs.gaugegroup.FullGaugeGroup(qutritMDL.dim)

    return qutritMDL
