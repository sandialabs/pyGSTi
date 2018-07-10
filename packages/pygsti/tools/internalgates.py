"""The standard unitaries used internally for compilers etc."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from . import gatetools as _gts

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