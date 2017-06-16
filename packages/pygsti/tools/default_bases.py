from basis import Basis, build_basis, get_conversion_mx

import numpy as _np

id2x2  = _np.array([[1,0],[0,1]])
sigmax = _np.array([[0,1],[1,0]])
sigmay = _np.array([[0,-1.0j],[1.0j,0]])
sigmaz = _np.array([[1,0],[0,-1]])

Pauli = Basis('pp', [('I', id2x2), ('X', sigmax), ('Y', sigmay), ('Z', sigmaz)], longname='Pauli')

e0 = _np.array([[1,0],[0,0]])
e1 = _np.array([[0,1],[0,0]])
e2 = _np.array([[0,0],[1,0]])
e3 = _np.array([[0,0],[0,1]])

Standard = Basis('std', [e0, e1, e2, e3], longname='Standard')

print(build_basis('pp'))
print(build_basis(Pauli))

print(get_conversion_mx(Pauli, Standard))
print(Pauli.is_normalized())
