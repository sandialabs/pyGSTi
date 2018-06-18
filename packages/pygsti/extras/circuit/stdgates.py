from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np

def standard_unitaries():
    
    std_unitaries = {}
    
    # The idle gate.
    std_unitaries['I'] = _np.array([[1.,0.],[0.,1.]],complex)

    # Non-idle single qubit gates
    std_unitaries['X'] = _np.array([[0.,1.],[1.,0.]],complex)
    std_unitaries['Y'] = _np.array([[0.,-1j],[1j,0.]],complex)
    std_unitaries['Z'] = _np.array([[1.,0.],[0.,-1.]],complex)    
    
    H = (1/_np.sqrt(2))*_np.array([[1.,1.],[1.,-1.]],complex) 
    P = _np.array([[1.,0.],[0.,1j]],complex)
    
    std_unitaries['H'] =  H  
    std_unitaries['P'] = P
    std_unitaries['HP'] = _np.dot(H,P)
    std_unitaries['PH'] = _np.dot(P,H)
    std_unitaries['HPH'] = _np.dot(H,_np.dot(P,H))
 
    # Two-qubit gates
    std_unitaries['CPHASE'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,-1.]],complex)
    std_unitaries['CNOT'] = _np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]],complex)
    std_unitaries['SWAP'] = _np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],complex)
    
    return std_unitaries
