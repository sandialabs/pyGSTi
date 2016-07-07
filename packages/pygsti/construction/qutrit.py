from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Routines for building qutrit gates and gate sets """

import numpy as _np
from scipy import linalg as _linalg

from ..tools import basistools as _bt
from ..      import objects    as _objs


#Define 2 qubit to symmetric (+) antisymmetric space transformation A:
A = _np.matrix([[1,0,0,0],
#               [0,0,0,1],
               [0,1./_np.sqrt(2),1./_np.sqrt(2),0],
               [0,1./_np.sqrt(2),-1./_np.sqrt(2),0],
               [0,0,0,1],])

X = _np.matrix([[0,1],[1,0]])
Y = _np.matrix([[0,-1j],[1j,0]])

#Returns X(theta)^\otimes 2
def X2qubit(theta):
    x = _np.matrix(_linalg.expm(-1j/2. * theta * _np.matrix([[0,1],[1,0]])))
    return _np.kron(x,x)

#Returns Y(theta)^\otimes 2
def Y2qubit(theta):
    y = _np.matrix(_linalg.expm(-1j/2. * theta * _np.matrix([[0,-1j],[1j,0]])))
    return _np.kron(y,y)

#Returns Molmer-Sorensen gate for two qubits
def ms2qubit(theta,phi):
    return _np.matrix(_linalg.expm(-1j/2 * theta *
                                 _np.kron(
                                    _np.cos(phi) * X + _np.sin(phi) * Y,
                                    _np.cos(phi) * X + _np.sin(phi) * Y)
                                 ))

#Projecting above gates into symmetric subspace (qutrit space)
#(state space ordering is |0> = |00>, |1> ~ |01>+|10>,|2>=|11>, so state |i> corresponds to i detector counts


#Removes columns and rows from inputArr
def _remove_from_matrix(inputArr, columns, rows, outputType = _np.matrix):
    inputArr = _np.array(inputArr)
    return outputType([
           [inputArr[row_num][col_num]
           for col_num in range(len(inputArr[row_num]))
           if not col_num in columns]

           for row_num in range(len(inputArr))
           if not row_num in rows])

def to_qutrit_space(inputMat):
    inputMat = _np.matrix(inputMat)
    return _remove_from_matrix(A * inputMat * A**-1,[2],[2])
#    return (A * inputMat * A**-1)[:3,:3]#Comment out above line and uncomment this line if you want the state space
#labelling to be |0>=|00>,|1>=|11>,|2>~|01>+|10>

def MS3(theta,phi):
    return to_qutrit_space(ms2qubit(theta,phi))

def XX3(theta):
    return to_qutrit_space(X2qubit(theta))

def YY3(theta):
    return to_qutrit_space(Y2qubit(theta))

def _random_rot(scale,arrType = _np.array, seed=None):
    rndm = _np.random.RandomState(seed)
    randH = scale * (rndm.randn(3,3) + 1j * rndm.randn(3,3))
    randH = _np.dot(_np.conj(randH.T), randH)
    randU = _linalg.expm(-1j * randH)
    return arrType(randU)

def make_qutrit_gateset(errorScale,Xangle = _np.pi/2, Yangle = _np.pi/2, MSglobal = _np.pi/2, MSlocal = 0, arrType = _np.array,similarity=False,seed=None):

    arrType = _np.array#Are we casting gates as matrices or arrays?

    rho0 = arrType(([[1,0,0],
                     [0,0,0],
                     [0,0,0]]))

    identity3 = arrType(_np.identity(3))
    identity3gm = _bt.std_to_gm(_np.reshape(identity3,(9,1)))

    E0 = arrType(_np.diag([1,0,0]))
    E1 = arrType(_np.diag([0,1,0]))
    E2 = arrType(_np.diag([0,0,1]))

    #Define gates as unitary ops on Hilbert space
    gateImx = arrType(identity3)
    gateXmx = arrType(XX3(Xangle))
    gateYmx = arrType(YY3(Yangle))
    gateMmx = arrType(MS3(MSglobal,MSlocal))

    #Now introduce unitary noise.

    scale = errorScale
    Xrand = _random_rot(scale,seed = seed)
    Yrand = _random_rot(scale)
    Mrand = _random_rot(scale)
    Irand = _random_rot(scale)


    if similarity:#Change basis for each gate; this preserves rotation angles, and should map identity to identity
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
    gateISO = _np.kron(_np.conj(gateImx),gateImx)
    gateISOgm = _bt.std_to_gm(gateISO)
    gateXSO = _np.kron(_np.conj(gateXmx),gateXmx)
    gateXSOgm = _bt.std_to_gm(gateXSO)
    gateYSO = _np.kron(_np.conj(gateYmx),gateYmx)
    gateYSOgm = _bt.std_to_gm(gateYSO)
    gateMSO = _np.kron(_np.conj(gateMmx),gateMmx)
    gateMSOgm = _bt.std_to_gm(gateMSO)


    rho0gm = _bt.std_to_gm(_np.reshape(rho0,(9,1)))
    E0gm =  _bt.std_to_gm(_np.reshape(E0,(9,1)))
    E1gm = _bt.std_to_gm(_np.reshape(E1,(9,1)))
    E2gm = _bt.std_to_gm(_np.reshape(E2,(9,1)))

    qutritGS = _objs.GateSet()
    qutritGS['rho0'] = rho0gm
    qutritGS['E0'] = E0gm
    qutritGS['E1'] = E1gm
    qutritGS['E2'] = E2gm
    qutritGS['identity'] = identity3gm
    qutritGS.spamdefs['0bright'] = ('rho0','E0')
    qutritGS.spamdefs['1bright'] = ('rho0','E1')
    qutritGS.spamdefs['2bright'] = ('rho0','E2')
    qutritGS['Gi'] = _objs.FullyParameterizedGate(arrType(gateISOgm))
    qutritGS['Gx'] = _objs.FullyParameterizedGate(arrType(gateXSOgm))
    qutritGS['Gy'] = _objs.FullyParameterizedGate(arrType(gateYSOgm))
    qutritGS['Gm'] = _objs.FullyParameterizedGate(arrType(gateMSOgm))
    return qutritGS
