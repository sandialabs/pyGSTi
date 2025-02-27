from pygsti.circuits.circuit import Circuit as C
from pygsti.baseobjs.label import Label as L

from qiskit import QuantumCircuit

def convert_qiskit_to_pygsti_circ(qc: QuantumCircuit) -> C:

    qubits = qc.qubits
    qubit_indices = []
    for qubit in qubits:
        qubit_indices.append(qubit._index)
    
    line_labels = [f'Q{index}' for index in qubit_indices]

    layer_indices = {}
    for line_label in line_labels:
        layer_indices[line_label] = 0

    instructions = qc.data

    pygsti_circ_layers = []
    layer_names = []

    for instruction in instructions:
        assert len(pygsti_circ_layers) == len(layer_names), "there must be one layer name for each layer!"

        # print(instruction)
        name = instruction.operation.name
        num_qubits = instruction.operation.num_qubits
        qubits = [f'Q{qubit._index}' for qubit in instruction.qubits]
        params = instruction.operation.params
        # print(name)
        # print(num_qubits)
        # print(params)

        if name == 'measure' or name == 'barrier':
            # print('skipping measure or barrier')
            continue

        label = convert_qiskit_instruction_to_pygsti_label(name, qubits, params)

        next_index = max(layer_indices[qubit] for qubit in qubits)
        for i in range(next_index, len(pygsti_circ_layers)):
            if name == layer_names[i]:
                # print(f'inserting gate {name} on qubits {qubits} in layer {i}')
                pygsti_circ_layers[i].append(label)
                for qubit in qubits:
                    layer_indices[qubit] = i + 1
                break

        else:
            # print(f'inserting gate {name} on qubits {qubits} in layer {len(pygsti_circ_layers)}')
            pygsti_circ_layers.append([label])
            layer_names.append(name)
            for qubit in qubits:
                layer_indices[qubit] = len(pygsti_circ_layers)
        
    circuit = C(pygsti_circ_layers, line_labels=line_labels)

    return circuit

def convert_qiskit_instruction_to_pygsti_label(name, qubits, params):
    num_qubits = len(qubits)

    if num_qubits == 1:
        qubit = qubits[0]
        if name == 'rz':
            label = L('Gzr', qubits, args=params)              
        elif name == 'sx':
            label = L('Gxpi2', qubits, args=params)
        elif name == 'x':
            label = L('Gxpi', qubits, args=params) # should this be Gc3 instead?
        elif name == 'u3':
            label = L('Gu3', qubits, args=params)
        elif name == 'delay':
            label = L('Gdelay', qubits, args=params)
        else:
            raise ValueError(f'cannot parse instruction {name} on qubits {qubits}! Please ensure that your qiskit circuit is expressed in a u3-cx gate set.')

    elif num_qubits == 2:
        if name == 'ecr': #NEED TO CHECK ORDERING
            label = L('Gecres', qubits, args=params) 
        elif name == 'cx':
            label = L('Gcnot', qubits, args=params) #tq and cq are reversed in the Instruction for some reason. Or are they?
        elif name == 'cz': #in qiskit, cphase admits an arbitary rotation. The pygsti cphase rotation angle is the same as the qiskit cz definition.
            label = L('Gcphase', qubits, args=params)
        else:
            raise ValueError('cannot parse this instruction!')
        
    else:
        raise ValueError('we only support 1 or 2 qubits at this point!')
    
    return label