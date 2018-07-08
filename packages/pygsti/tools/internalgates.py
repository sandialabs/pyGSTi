"""The standard unitaries used internally for compilers etc."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from . import gatetools as _gts


std_unitaries = {}
std_unitaries['I'] = _np.array([[1,0],[0,1]],complex)
std_unitaries['X'] = _np.array([[0,1],[1,0]],complex)
std_unitaries['Y'] = _np.array([[0,-1.0j],[1.0j,0]],complex)
std_unitaries['Z'] = _np.array([[1,0],[0,-1]],complex)
  
std_unitaries['H'] = (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex)  
std_unitaries['P'] = _np.array([[1.,0.],[0.,1j]],complex)
std_unitaries['HP'] = _np.dot(std_unitaries['H'],std_unitaries['P'])
std_unitaries['PH'] = _np.dot(std_unitaries['P'],std_unitaries['H'])
std_unitaries['HPH'] = _np.dot(std_unitaries['H'],_np.dot(std_unitaries['P'],std_unitaries['H']))

# Tim todo : add the 1-qubit Clifford group.

# Two-qubit gates
# This is not currently used anywhere, so commented out.
std_unitaries['CPHASE'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
std_unitaries['CNOT'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
# This is not currently used anywhere I think, so commented out
std_unitaries['SWAP'] = _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)

# def get_standard_unitary():
# 	std_unitaries = {}
# 	std_unitaries['I'] = _np.array([[1,0],[0,1]],complex)
# 	std_unitaries['X'] = _np.array([[0,1],[1,0]],complex)
# 	std_unitaries['Y'] = _np.array([[0,-1.0j],[1.0j,0]],complex)
# 	std_unitaries['Z'] = _np.array([[1,0],[0,-1]],complex)
	  
# 	std_unitaries['H'] = (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex)  
# 	std_unitaries['P'] = _np.array([[1.,0.],[0.,1j]],complex)
# 	std_unitaries['HP'] = _np.dot(std_unitaries['H'],std_unitaries['P'])
# 	std_unitaries['PH'] = _np.dot(std_unitaries['P'],std_unitaries['H'])
# 	std_unitaries['HPH'] = _np.dot(std_unitaries['H'],_np.dot(P,std_unitaries['H']))

# 	# Tim todo : add the 1-qubit Clifford group.

# 	# Two-qubit gates
# 	# This is not currently used anywhere, so commented out.
# 	std_unitaries['CPHASE'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
# 	std_unitaries['CNOT'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
# 	# This is not currently used anywhere I think, so commented out
# 	std_unitaries['SWAP'] = _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)

# 	return std_unitaries

# def get_standard_unitary(gate_name):
# 	if gate_name == 'I': return _np.array([[1,0],[0,1]],complex)
# 	elif gate_name == 'X': return _np.array([[0,1],[1,0]],complex)
# 	elif gate_name == 'Y': return _np.array([[0,-1.0j],[1.0j,0]],complex)
# 	elif gate_name == 'Z': return  _np.array([[1,0],[0,-1]],complex)
	  
# 	elif gate_name == 'H': return (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex)  
# 	elif gate_name == 'P': return _np.array([[1.,0.],[0.,1j]],complex)
# 	elif gate_name == 'HP': return _np.dot((1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex) ,_np.array([[1.,0.],[0.,1j]],complex))
# 	elif gate_name == 'PH': return _np.dot(_np.array([[1.,0.],[0.,1j]],complex),(1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex) )
# 	elif gate_name == 'HPH': return return_np.dot(_np.array([[1.,1.],[1.,-1.]],complex) ,_np.dot(P,_np.array([[1.,1.],[1.,-1.]],complex) ))/2

# 	# Tim todo : add the 1-qubit Clifford group.

# 	# Two-qubit gates
# 	# This is not currently used anywhere, so commented out.
# 	elif gate_name == 'CPHASE': return _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
# 	elif gate_name == 'CNOT': return _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
# 	# This is not currently used anywhere I think, so commented out
# 	elif gate_name == 'SWAP': return _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)
# 	else:
# 		raise ValueError("Input is not a standard unitary!")

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

def is_gate_pauli_equivalent_to_this_standard_unitary(gate_unitary,standard_gate_name):
	return