from pygsti.extras.idletomography import idtcore
import numpy as np
from pygsti.baseobjs import Basis
from itertools import product, permutations
from pygsti.baseobjs import basisconstructors

np.set_printoptions(precision=4, linewidth=1000, suppress=True)

pauli_list = basisconstructors.pp_matrices_dict(2, normalize=False)

hamiltonian_jacobian_coefs = idtcore.build_class_jacobian("H", 1)
print(hamiltonian_jacobian_coefs)
idtcore.dict_to_jacobian(hamiltonian_jacobian_coefs, "H", 1)
stochastic_jacobian_coefs = idtcore.build_class_jacobian("S", 1)
print(stochastic_jacobian_coefs)
correlation_jacobian_coefs = idtcore.build_class_jacobian("C", 1)
print(correlation_jacobian_coefs)
anti_symmetric_jacobian_coefs = idtcore.build_class_jacobian("A", 1)
print(anti_symmetric_jacobian_coefs)


full_jacobian = np.hstack(
    (
        hamiltonian_jacobian,
        stochastic_jacobian,
        correlation_jacobian,
        anti_symmetric_jacobian,
    ),
)
print("Hey full Jacobian:\n", full_jacobian)
inverse_jacobian = np.linalg.pinv(full_jacobian)
print(inverse_jacobian)
quit()

blah = idtcore.jacobian_index_label(1)
print(blah)

import pygsti
from pygsti.extras import idletomography as idt

n_qubits = 1
gates = ["Gi", "Gx", "Gy", "Gcnot"]
max_lengths = [1, 2, 4, 8]
pspec = pygsti.processors.QubitProcessorSpec(
    n_qubits, gates, geometry="line", nonstd_gate_unitaries={(): 1}
)

mdl_target = pygsti.models.create_crosstalk_free_model(pspec)
paulidicts = idt.determine_paulidicts(mdl_target)
