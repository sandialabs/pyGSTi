import pygsti
import numpy as _np
import copy as _copy

from pygsti.algorithms import compilers as _cmpl
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti import tools as _tools
from pygsti.tools import group as _rbobjs
from pygsti.tools import symplectic as _symp
from pygsti.algorithms import randomcircuit as _rc
from pygsti.tools import compilationtools as _comp

from pygsti.protocols import protocol as _proto
from pygsti.algorithms import rbfit as _rbfit
from pygsti.algorithms import mirroring as _mirroring

from pygsti.protocols import rb as _rb
from pygsti.protocols import vb as _vb


from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR

def create_random_quintuple_layered_circuit(pspec, qubit_labels, length = 1, two_q_gate_density = .25, one_q_gate_names = 'all', pdist = 'uniform', modelname = 'clifford', rand_state = None):
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.num_qubits
        qubits = pspec.qubit_labels[:]  # copy this list
        
    circuit = _cir.Circuit([], line_labels = qubit_labels, editable = True)
    for i in range(length):
        layer = sample_quint_layer(pspec, qubit_labels, two_q_gate_density = two_q_gate_density, one_q_gate_names = one_q_gate_names, rand_state = rand_state, pdist = pdist, modelname = modelname)
        c = _cir.Circuit(layer, line_labels = qubit_labels)
        circuit.append_circuit_inplace(c)
    circuit.done_editing()
    return circuit

def sample_quint_layer(pspec, qubit_labels, two_q_gate_density = .25, gate_args_lists = None, rand_state = None, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all'):
    layer_1 = _rc.sample_circuit_layer_of_one_q_gates(pspec, qubit_labels, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all', rand_state=rand_state)
    layer_2 = [[]]
    mid_layer = _rc.sample_circuit_layer_by_edgegrab(pspec, qubit_labels, two_q_gate_density = two_q_gate_density, rand_state=rand_state)
    layer_4 = [[]]
    layer_5 = _rc.sample_circuit_layer_of_one_q_gates(pspec, qubit_labels, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all', rand_state=rand_state)
    return [layer_1, layer_2, mid_layer, layer_4, layer_5]
  

