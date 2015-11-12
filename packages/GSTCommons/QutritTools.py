import numpy as np
import GST
import scipy
from scipy import linalg
from numpy import pi
from GSTCommons import MakeLists_WholeGermPowers
import matplotlib
import time

#Define 2 qubit to symmetric (+) antisymmetric space transformation A:
A = np.matrix([[1,0,0,0],
#               [0,0,0,1],
               [0,1./np.sqrt(2),1./np.sqrt(2),0],
               [0,1./np.sqrt(2),-1./np.sqrt(2),0],
               [0,0,0,1],])

X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])

#Returns X(theta)^\otimes 2
def X2qubit(theta):
    x = np.matrix(linalg.expm(-1j/2. * theta * np.matrix([[0,1],[1,0]])))
    return np.kron(x,x)

#Returns Y(theta)^\otimes 2
def Y2qubit(theta):
    y = np.matrix(linalg.expm(-1j/2. * theta * np.matrix([[0,-1j],[1j,0]])))
    return np.kron(y,y)

#Returns Molmer-Sorensen gate for two qubits
def MS2qubit(theta,phi):
    return np.matrix(linalg.expm(-1j/2 * theta * 
                                 np.kron(
                                    np.cos(phi) * X + np.sin(phi) * Y,
                                    np.cos(phi) * X + np.sin(phi) * Y)
                                 ))

#Projecting above gates into symmetric subspace (qutrit space) 
#(state space ordering is |0> = |00>, |1> ~ |01>+|10>,|2>=|11>, so state |i> corresponds to i detector counts


#Removes columns and rows from inputArr
def remove_from_matrix(inputArr, columns, rows, outputType = np.matrix):
    inputArr = np.array(inputArr)
    return outputType([
           [inputArr[row_num][col_num]
           for col_num in range(len(inputArr[row_num])) 
           if not col_num in columns]

           for row_num in range(len(inputArr))
           if not row_num in rows])

def toQutritSpace(inputMat):
    inputMat = np.matrix(inputMat)
    return remove_from_matrix(A * inputMat * A**-1,[2],[2])
#    return (A * inputMat * A**-1)[:3,:3]#Comment out above line and uncomment this line if you want the state space
#labelling to be |0>=|00>,|1>=|11>,|2>~|01>+|10>

def MS3(theta,phi):
    return toQutritSpace(MS2qubit(theta,phi))

def XX3(theta):
    return toQutritSpace(X2qubit(theta))

def YY3(theta):
    return toQutritSpace(Y2qubit(theta))

def randomRot(scale,arrType = np.array, seed=None):
    if seed is not None:
        np.random.seed(seed)
    randH = scale * (np.random.randn(3,3) + 1j * np.random.randn(3,3))
    randH = np.dot(np.conj(randH.T), randH)
    randU = scipy.linalg.expm(-1j * randH)
    return arrType(randU)
    
def makeQutritGS(errorScale,Xangle = np.pi/2, Yangle = np.pi/2, MSglobal = np.pi/2, MSlocal = 0, arrType = np.array,similarity=False,seed=None):

    arrType = np.array#Are we casting gates as matrices or arrays?

    rho0 = arrType(([[1,0,0],
                     [0,0,0],
                     [0,0,0]]))

    identity3 = arrType(np.identity(3))
    identity3gm = GST.BasisTools.basisChg_StdToGellMann(np.reshape(identity3,(9,1)))

    E0 = arrType(np.diag([1,0,0]))
    E1 = arrType(np.diag([0,1,0]))
    E2 = arrType(np.diag([0,0,1]))

    #Define gates as unitary ops on Hilbert space
    gateImx = arrType(identity3)
    gateXmx = arrType(XX3(Xangle))
    gateYmx = arrType(YY3(Yangle))
    gateMmx = arrType(MS3(MSglobal,MSlocal))

    #Now introduce unitary noise.

    if seed is not None:
        np.random.seed(seed)
    scale = errorScale
    Xrand = randomRot(scale,seed = seed)
    Yrand = randomRot(scale)
    Mrand = randomRot(scale)
    Irand = randomRot(scale)


    if similarity:#Change basis for each gate; this preserves rotation angles, and should map identity to identity
        gateXmx = np.dot(np.dot(np.conj(Xrand).T, gateXmx), Xrand)
        gateYmx = np.dot(np.dot(np.conj(Yrand).T, gateYmx), Yrand)
        gateMmx = np.dot(np.dot(np.conj(Mrand).T, gateMmx), Mrand)
        gateImx = np.dot(np.dot(np.conj(Irand).T, gateMmx), Irand)

    else:
        gateXmx = np.dot(gateXmx, Xrand)
        gateYmx = np.dot(gateYmx, Yrand)
        gateMmx = np.dot(gateMmx, Mrand)
        gateImx = np.dot(gateImx, Mrand)

    #Change gate representation to superoperator in Gell-Mann basis
    gateISO = np.kron(np.conj(gateImx),gateImx)
    gateISOgm = GST.BasisTools.basisChg_StdToGellMann(gateISO)
    gateXSO = np.kron(np.conj(gateXmx),gateXmx)
    gateXSOgm = GST.BasisTools.basisChg_StdToGellMann(gateXSO)
    gateYSO = np.kron(np.conj(gateYmx),gateYmx)
    gateYSOgm = GST.BasisTools.basisChg_StdToGellMann(gateYSO)
    gateMSO = np.kron(np.conj(gateMmx),gateMmx)
    gateMSOgm = GST.BasisTools.basisChg_StdToGellMann(gateMSO)


    rho0gm = GST.BasisTools.basisChg_StdToGellMann(np.reshape(rho0,(9,1)))
    E0gm =  GST.BasisTools.basisChg_StdToGellMann(np.reshape(E0,(9,1)))
    E1gm = GST.BasisTools.basisChg_StdToGellMann(np.reshape(E1,(9,1)))
    E2gm = GST.BasisTools.basisChg_StdToGellMann(np.reshape(E2,(9,1)))

    qutritGS = GST.GateSet()
    qutritGS.set_rhoVec(rho0gm)
    qutritGS.set_EVec(E0gm,0)
    qutritGS.set_EVec(E1gm,1)
    qutritGS.set_EVec(E2gm,2)
    qutritGS.set_identityVec(identity3gm)
    qutritGS.add_SPAM_label(0,0,'0bright')
    qutritGS.add_SPAM_label(0,1,'1bright')
    qutritGS.add_SPAM_label(0,2,'2bright')
    qutritGS.makeSPAMs()
    qutritGS.set_gate('Gi',GST.Gate.FullyParameterizedGate(arrType(gateISOgm)))
    qutritGS.set_gate('Gx',GST.Gate.FullyParameterizedGate(arrType(gateXSOgm)))
    qutritGS.set_gate('Gy',GST.Gate.FullyParameterizedGate(arrType(gateYSOgm)))
    qutritGS.set_gate('Gm',GST.Gate.FullyParameterizedGate(arrType(gateMSOgm)))
    return qutritGS