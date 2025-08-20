"""
Routines for building qutrit gates and models
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy import linalg as _linalg

from pygsti.baseobjs import Basis as _Basis, statespace as _statespace
from pygsti.models.gaugegroup import TPGaugeGroup as _TPGaugeGroup
from pygsti.modelmembers.operations import FullTPOp as _FullTPOp
from pygsti.modelmembers.povms import TPPOVM as _TPPOVM
from pygsti.modelmembers.states import TPState as _TPState
from pygsti.models import ExplicitOpModel as _ExplicitOpModel
from pygsti.tools import unitary_to_superop, change_basis

#Define 2 qubit to symmetric (+) antisymmetric space transformation A:
A = _np.matrix([[1, 0, 0, 0],
                #               [0,0,0,1],
                [0, 1. / _np.sqrt(2), 1. / _np.sqrt(2), 0],
                [0, 1. / _np.sqrt(2), -1. / _np.sqrt(2), 0],
                [0, 0, 0, 1], ])

X = _np.array([[0, 1], [1, 0]])
Y = _np.array([[0, -1j], [1j, 0]])


def _x_2qubit(theta):
    r"""
    Returns X(theta)^\otimes 2 (2-qubit 'XX' unitary)

    Parameters
    ----------
    theta : float
        rotation angle: U = exp(-i/2 * theta * sigmaX)

    Returns
    -------
    numpy.ndarray
    """
    x = _np.array(_linalg.expm(-1j / 2. * theta * _np.array([[0, 1], [1, 0]])))
    return _np.kron(x, x)


def _y_2qubit(theta):
    r"""
    Returns Y(theta)^\otimes 2 (2-qubit 'YY' unitary)

    Parameters
    ----------
    theta : float
        rotation angle: U = exp(-i/2 * theta * sigmaY)

    Returns
    -------
    numpy.ndarray
    """
    y = _np.array(_linalg.expm(-1j / 2. * theta * _np.array([[0, -1j], [1j, 0]])))
    return _np.kron(y, y)


def _ms_2qubit(theta, phi):
    """
    Returns Molmer-Sorensen gate for two qubits

    Returns the unitary given by:
    `U = exp(i/2 * theta * A otimes A)` where
    `A = cos(phi)*sigmaX + sin(phi)*sigmaY`

    Parameters
    ----------
    theta : float
        global rotation angle

    phi : float
        local rotation angle

    Returns
    -------
    numpy.ndarray
    """
    return _np.array(_linalg.expm(-1j / 2 * theta
                                   * _np.kron(
                                       _np.cos(phi) * X + _np.sin(phi) * Y,
                                       _np.cos(phi) * X + _np.sin(phi) * Y)
                                   ))

#Projecting above gates into symmetric subspace (qutrit space)
#(state space ordering is |0> = |00>, |1> ~ |01>+|10>,|2>=|11>, so state |i> corresponds to i detector counts


#Removes columns and rows from input_arr
def _remove_from_matrix(input_arr, columns, rows, output_type=_np.array):
    input_arr = _np.array(input_arr)
    return output_type([
        [input_arr[row_num][col_num]
            for col_num in range(len(input_arr[row_num]))
            if col_num not in columns]

        for row_num in range(len(input_arr))
        if row_num not in rows])


def to_qutrit_space(input_mat):
    """
    Projects a 2-qubit unitary matrix onto the symmetric "qutrit space"

    Parameters
    ----------
    input_mat : numpy.ndarray
        the unitary matrix to project.

    Returns
    -------
    numpy.ndarray
    """
    input_mat = _np.array(input_mat)
    return _remove_from_matrix(A * input_mat * _np.linalg.inv(A), [2], [2])
#    return (A * input_mat * A**-1)[:3,:3]#Comment out above line and uncomment this line if you want the state space
#labelling to be |0>=|00>,|1>=|11>,|2>~|01>+|10>


def _ms_qutrit(theta, phi):
    """
    Returns Qutrit Molmer-Sorenson unitary on the qutrit space

    Parameters
    ----------
    theta : float
        rotation angle

    phi : float
        rotation angle

    Returns
    -------
    numpy.ndarray
    """
    return to_qutrit_space(_ms_2qubit(theta, phi))


def _xx_qutrit(theta):
    """
    Returns Qutrit XX unitary

    Parameters
    ----------
    theta : float
        rotation angle.

    Returns
    -------
    numpy.ndarray
    """
    return to_qutrit_space(_x_2qubit(theta))


def _yy_qutrit(theta):
    """
    Returns Qutrit YY unitary

    Parameters
    ----------
    theta : float
        rotation angle

    Returns
    -------
    numpy.ndarray
    """
    return to_qutrit_space(_y_2qubit(theta))


def _random_rot(scale, rand_state, arr_type=_np.array):
    randH = scale * (rand_state.randn(3, 3) + 1j * rand_state.randn(3, 3))
    randH = _np.dot(_np.conj(randH.T), randH)
    randU = _linalg.expm(-1j * randH)
    return arr_type(randU)


def create_qutrit_model(error_scale, x_angle=_np.pi / 2, y_angle=_np.pi / 2,
                        ms_global=_np.pi / 2, ms_local=0,
                        similarity=False, seed=None, basis='qt', evotype='default'):
    """
    Constructs a standard qutrit :class:`Model`.

    This model contains the identity, XX, YY, and Molmer-Sorenson gates.

    Parameters
    ----------
    error_scale : float
        Magnitude of random rotations to apply to the returned model.  If
        zero, then perfect "ideal" gates are constructed.

    x_angle : float, optional
        The rotation angle of each X in the XX gate.

    y_angle : float, optional
        The rotation angle of each Y in the YY gate.

    ms_global : float, optional
        The global Molmer-Sorenson angle (theta)

    ms_local : float, optional
        The local Molmer-Sorenson angle (theta)

    similarity : bool, optional
        If true, then apply the random rotations (whose strengths are given
        by `error_scale`) as similarity transformations rather than just as
        post-multiplications to the ideal operation matrices.

    seed : int, optional
        The seed used to generate random rotations.

    basis : str, optional
        The string abbreviation of the basis of the returned vector.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm) and Qutrit (qt).  A `Basis`
        object may also be used.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

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
    gateXmx = arrType(_xx_qutrit(x_angle))
    gateYmx = arrType(_yy_qutrit(y_angle))
    gateMmx = arrType(_ms_qutrit(ms_global, ms_local))

    #Now introduce unitary noise.
    scale = error_scale
    rndm = _np.random.RandomState(seed)
    Xrand = _random_rot(scale, rndm)
    Yrand = _random_rot(scale, rndm)
    Mrand = _random_rot(scale, rndm)
    Irand = _random_rot(scale, rndm)

    if similarity:  # Change basis for each gate; this preserves rotation angles, and should map identity to identity
        gateXmx = _np.dot(_np.dot(_np.conj(Xrand).T, gateXmx), Xrand)
        gateYmx = _np.dot(_np.dot(_np.conj(Yrand).T, gateYmx), Yrand)
        gateMmx = _np.dot(_np.dot(_np.conj(Mrand).T, gateMmx), Mrand)
        gateImx = _np.dot(_np.dot(_np.conj(Irand).T, gateImx), Irand)

    else:
        gateXmx = _np.dot(gateXmx, Xrand)
        gateYmx = _np.dot(gateYmx, Yrand)
        gateMmx = _np.dot(gateMmx, Mrand)
        gateImx = _np.dot(gateImx, Irand)

    #Change gate representation to superoperator in Gell-Mann basis
    gateISOfinal = unitary_to_superop(gateImx, basis)
    gateXSOfinal = unitary_to_superop(gateXmx, basis)
    gateYSOfinal = unitary_to_superop(gateYmx, basis)
    gateMSOfinal = unitary_to_superop(gateMmx, basis)

    rho0final = change_basis(_np.reshape(rho0, (9, 1)), "std", basis)
    E0final = change_basis(_np.reshape(E0, (9, 1)), "std", basis)
    E1final = change_basis(_np.reshape(E1, (9, 1)), "std", basis)
    E2final = change_basis(_np.reshape(E2, (9, 1)), "std", basis)

    state_space = _statespace.ExplicitStateSpace(['QT'], [3])
    qutritMDL = _ExplicitOpModel(state_space, _Basis.cast(basis, 9), evotype=evotype)
    qutritMDL.preps['rho0'] = _TPState(rho0final, evotype=evotype)
    qutritMDL.povms['Mdefault'] = _TPPOVM([('0bright', E0final),
                                           ('1bright', E1final),
                                           ('2bright', E2final)], evotype=evotype)
    qutritMDL.operations['Gi', 'QT'] = _FullTPOp(arrType(gateISOfinal), basis, evotype, state_space)
    qutritMDL.operations['Gx', 'QT'] = _FullTPOp(arrType(gateXSOfinal), basis, evotype, state_space)
    qutritMDL.operations['Gy', 'QT'] = _FullTPOp(arrType(gateYSOfinal), basis, evotype, state_space)
    qutritMDL.operations['Gm', 'QT'] = _FullTPOp(arrType(gateMSOfinal), basis, evotype, state_space)
    qutritMDL.default_gauge_group = _TPGaugeGroup(state_space, qutritMDL.basis, evotype)

    return qutritMDL
