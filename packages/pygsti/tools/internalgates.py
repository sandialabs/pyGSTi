"""The standard unitaries used internally for compilers etc."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from . import gatetools as _gts

def get_internal_gatename_unitaries():

	"""
	todo : docstring.
	"""

	# A dictionary containing the "standard" unitaries understood by pyGSTi. This dictionary is what defines
	# these unitaries, and all other code should refer to this for their definitions. Note that some of the
	# gates have more than one name. E.g., the 1-qubit Clifford gates named as C0, C1, to, C23 contain the
	# other 1-qubit Clifford gates (i.e, 'I' and 'C0' are equal).
	std_unitaries = {}
	# The 1-qubit Paulis
	std_unitaries['I'] = _np.array([[1,0],[0,1]],complex)
	std_unitaries['X'] = _np.array([[0,1],[1,0]],complex)
	std_unitaries['Y'] = _np.array([[0,-1.0j],[1.0j,0]],complex)
	std_unitaries['Z'] = _np.array([[1,0],[0,-1]],complex)
	# 5 gates constructed from Hadamard and Phase which each represent 1 of the 5 1-qubit Clifford gate classes
	# that cannot be converted to each other or the identity via Pauli operators.
	std_unitaries['H'] = (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex)  
	std_unitaries['P'] = _np.array([[1.,0.],[0.,1j]],complex)
	std_unitaries['HP'] = _np.dot(std_unitaries['H'],std_unitaries['P'])
	std_unitaries['PH'] = _np.dot(std_unitaries['P'],std_unitaries['H'])
	std_unitaries['HPH'] = _np.dot(std_unitaries['H'],_np.dot(std_unitaries['P'],std_unitaries['H']))
	# The 1-qubit Clifford group. The labelling is the same as in the the 1-qubit Clifford group generated
	# in pygsti.extras.rb.group, with the mapping 'Ci' - > 'Gci'. (we keep with the convention here of not have
	# hard-coded unitaries starting with a 'G'.)
	std_unitaries['C0'] =  _np.array([[1,0],[0,1]],complex)
	std_unitaries['C1'] =  _np.array([[1,-1j],[1,1j]],complex)/_np.sqrt(2)
	std_unitaries['C2'] = _np.array([[1,1],[1j,-1j]],complex)/_np.sqrt(2)
	std_unitaries['C3'] = _np.array([[0,1],[1,0]],complex)
	std_unitaries['C4'] = _np.array([[-1,-1j],[1,-1j]],complex)/_np.sqrt(2)
	std_unitaries['C5'] = _np.array([[1,1],[-1j,1j]],complex)/_np.sqrt(2)
	std_unitaries['C6'] = _np.array([[0,-1j],[1j,0]],complex)
	std_unitaries['C7'] = _np.array([[1j,1],[-1j,1]],complex)/_np.sqrt(2)
	std_unitaries['C8'] = _np.array([[1j,-1j],[1,1]],complex)/_np.sqrt(2)
	std_unitaries['C9'] = _np.array([[1,0],[0,-1]],complex)
	std_unitaries['C10'] = _np.array([[1,1j],[1,-1j]],complex)/_np.sqrt(2)
	std_unitaries['C11'] = _np.array([[1,-1],[1j,1j]],complex)/_np.sqrt(2)
	std_unitaries['C12'] = _np.array([[1,1],[1,-1]],complex)/_np.sqrt(2)
	std_unitaries['C13'] = _np.array([[0.5-0.5j,0.5+0.5j],[0.5+0.5j,0.5-0.5j]],complex)
	std_unitaries['C14'] = _np.array([[1,0],[0,1j]],complex)
	std_unitaries['C15'] = _np.array([[1,1],[-1,1]],complex)/_np.sqrt(2)
	std_unitaries['C16'] = _np.array([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]],complex)
	std_unitaries['C17'] = _np.array([[0,1],[1j,0]],complex)
	std_unitaries['C18'] = _np.array([[1j,-1j],[-1j,-1j]],complex)/_np.sqrt(2)
	std_unitaries['C19'] = _np.array([[0.5+0.5j,-0.5+0.5j],[0.5-0.5j,-0.5-0.5j]],complex)
	std_unitaries['C20'] = _np.array([[0,-1j],[-1,0]],complex)
	std_unitaries['C21'] = _np.array([[1,-1],[1,1]],complex)/_np.sqrt(2)
	std_unitaries['C22'] = _np.array([[0.5+0.5j,0.5-0.5j],[-0.5+0.5j,-0.5-0.5j]],complex)
	std_unitaries['C23'] = _np.array([[1,0],[0,-1j]],complex)
	# Standard 2-qubit gates.
	std_unitaries['CPHASE'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
	std_unitaries['CNOT'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
	std_unitaries['SWAP'] = _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)

	return std_unitaries

def is_gate_this_standard_unitary(gate_unitary,standard_gate_name):
	"""
	Returns True if a unitary is the specified standard gate, up to phase.
	Else, returns false.
	"""
	if _np.shape(gate_unitary) != _np.shape(std_unitaries[standard_gate_name]):
		return False
	else:
		pm_input = _gts.unitary_to_pauligate(gate_unitary)
		pm_std = _gts.unitary_to_pauligate(std_unitaries[standard_gate_name])
		equal = _np.allclose(pm_input,pm_std)
		return equal

# Currently not needed, but might be added in.
#
#def is_gate_pauli_equivalent_to_this_standard_unitary(gate_unitary,standard_gate_name):
#	return
#

def get_standard_gatename_unitaries():
    """
    Constructs and returns a dictionary of unitary matrices describing the
    action of "standard" gates.  These gates (also the keys of the returned
    dictionary) are:

    - 'Gi' : the 1Q idle operation
    - 'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
    - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations around X, Y and Z.
    - 'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
     'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
    - 'Gh' : Hadamard
    - 'Gp', 'Gpdag' : phase and inverse phase (an alternative notation/name for Gzpi and Gzmpi2)
    - 'Gcphase','Gcnot','Gswap' : standard 2Q gates

    Returns
    -------
    dict of numpy.ndarray objects.
    """
    std_unitaries = {}

    sigmax = _np.array([[0,1],[1,0]])
    sigmay = _np.array([[0,-1.0j],[1.0j,0]])
    sigmaz = _np.array([[1,0],[0,-1]])
    def Ugate(exp):
        return _np.array(_spl.expm(-1j * exp/2),complex)
    
    std_unitaries['Gi'] = _np.array([[1.,0.],[0.,1.]],complex)

    std_unitaries['Gxpi2'] = Ugate(_np.pi/2 * sigmax)
    std_unitaries['Gypi2'] = Ugate(_np.pi/2 * sigmay)
    std_unitaries['Gzpi2'] = Ugate(_np.pi/2 * sigmaz)

    std_unitaries['Gxpi'] = _np.array([[0.,1.],[1.,0.]],complex)
    std_unitaries['Gypi'] = _np.array([[0.,-1j],[1j,0.]],complex)
    std_unitaries['Gzpi'] = _np.array([[1.,0.],[0.,-1.]],complex)  

    std_unitaries['Gxmpi2'] = Ugate(-1*_np.pi/2 * sigmax)
    std_unitaries['Gympi2'] = Ugate(-1*_np.pi/2 * sigmay)
    std_unitaries['Gzmpi2'] = Ugate(-1*_np.pi/2 * sigmaz)
    
    H = (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex) 
    P = _np.array([[1.,0.],[0.,1j]],complex)
    Pdag = _np.array([[1.,0.],[0.,-1j]],complex)
    
    std_unitaries['Gh'] =  H  
    std_unitaries['Gp'] = P
    std_unitaries['Gpdag'] = Pdag
    std_unitaries['Ghp'] = _np.dot(H,P)
    std_unitaries['Gph'] = _np.dot(P,H)
    std_unitaries['Ghph'] = _np.dot(H,_np.dot(P,H))

    std_unitaries['Gt'] = _np.array([[1.,0.],[0.,_np.exp(1j*_np.pi/4)]],complex)
    std_unitaries['Gtdag'] =_np.array([[1.,0.],[0.,_np.exp(-1j*_np.pi/4)]],complex)
 
    # Two-qubit gates
    std_unitaries['Gcphase'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
    std_unitaries['Gcnot'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
    std_unitaries['Gswap'] = _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)
    
    return std_unitaries

def get_standard_gatenames_quil_conversions():
	"""
	docstring todo (tim). Currently there are gates with standard gate names with 
	no conversion to quil.
	"""
	std_gatenames_to_quil = {}
	std_gatenames_to_quil['Gi'] = 'I'
	std_gatenames_to_quil['Gxpi2'] = 'RX(pi/2)'
	std_gatenames_to_quil['Gxmpi2'] = 'RX(-pi/2)'
	std_gatenames_to_quil['Gxpi'] = 'X'
	std_gatenames_to_quil['Gzpi2'] = 'RZ(pi/2)'
	std_gatenames_to_quil['Gzmpi2'] = 'RZ(-pi/2)'
	std_gatenames_to_quil['Gzpi'] = 'Z'
	std_gatenames_to_quil['Gypi'] = 'Y'
	std_gatenames_to_quil['Gp'] = 'RZ(pi/2)' # todo : check that this is correct, and shouldn't instead be -pi/2
	std_gatenames_to_quil['Gpdag'] = 'RZ(-pi/2)' # todo : check that this is correct, and shouldn't instead be +pi/2
	std_gatenames_to_quil['Gcphase'] = 'CZ'
	std_gatenames_to_quil['Gcnot'] = 'CNOT'

	return std_gatenames_to_quil