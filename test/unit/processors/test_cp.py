from ..util import BaseCase

import numpy as np

from pygsti.baseobjs.label import Label as L
from pygsti.circuits.circuit import Circuit as C
from pygsti.processors.random_compilation import RandomCompilation
from pygsti.tools.internalgates import Gu3, standard_gatename_unitaries as _standard_gatename_unitaries


def get_clifford_from_unitary(U):
    clifford_unitaries = {k: v for k, v in _standard_gatename_unitaries().items()
                          if 'Gc' in k and v.shape == (2, 2)}
    for k,v in clifford_unitaries.items():
        for phase in [1, -1, 1j, -1j]:
            if np.allclose(U, phase*v):
                return k
            
    raise RuntimeError(f'Failed to look up Clifford for unitary:\n{U}')

class TestCentralPauli(BaseCase):

    def test_u3_x_errors_parallel(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0]), #I
                L('Gu3', ['Q2'], args=[np.pi,0,np.pi]), #X
                L('Gu3', ['Q3'], args=[np.pi,np.pi/2,np.pi/2]), #Y
                L('Gu3', ['Q4'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([0,0,0,0,2,2,2,2,]) # all X errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # This is passing but with (global) phase differences

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc3','Gc3','Gc3','Gc3'],['Gc0', 'Gc3', 'Gc6', 'Gc9']])

    def test_u3_z_errors_parallel(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0]), #I
                L('Gu3', ['Q2'], args=[np.pi,0,np.pi]), #X
                L('Gu3', ['Q3'], args=[np.pi,np.pi/2,np.pi/2]), #Y
                L('Gu3', ['Q4'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,2,2,0,0,0,0]) # all Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc9','Gc9','Gc9','Gc9'],['Gc0', 'Gc3', 'Gc6', 'Gc9']])

    def test_u3_x_and_z_errors_parallel(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0]), #I
                L('Gu3', ['Q2'], args=[np.pi,0,np.pi]), #X
                L('Gu3', ['Q3'], args=[np.pi,np.pi/2,np.pi/2]), #Y
                L('Gu3', ['Q4'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,2,2,2,2,2,2]) # all X and Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc6','Gc6','Gc6','Gc6'],['Gc0', 'Gc3', 'Gc6', 'Gc9']])

    def test_u3_x_errors_serial(self):
        line_labels = ['Q1']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0])], #I
                [L('Gu3', ['Q1'], args=[np.pi,0,np.pi])], #X
                [L('Gu3', ['Q1'], args=[np.pi,np.pi/2,np.pi/2])], #Y
                [L('Gu3', ['Q1'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([0,2]) # all X errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        # print(rc_circ)

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc3'], ['Gc0'], ['Gc3'], ['Gc6'], ['Gc9']])

    def test_u3_z_errors_serial(self):
        line_labels = ['Q1']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0])], #I
                [L('Gu3', ['Q1'], args=[np.pi,0,np.pi])], #X
                [L('Gu3', ['Q1'], args=[np.pi,np.pi/2,np.pi/2])], #Y
                [L('Gu3', ['Q1'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,0]) # all Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        # print(rc_circ)

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc9'], ['Gc0'], ['Gc3'], ['Gc6'], ['Gc9']])

    def test_u3_x_and_z_errors_serial(self):
        line_labels = ['Q1']
        layers = [[L('Gu3', ['Q1'], args=[0,0,0])], #I
                [L('Gu3', ['Q1'], args=[np.pi,0,np.pi])], #X
                [L('Gu3', ['Q1'], args=[np.pi,np.pi/2,np.pi/2])], #Y
                [L('Gu3', ['Q1'], args=[0,0,np.pi])]] #Z

        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2]) # all X and Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, _] = rc.compile(circ, test_layer)

        # print(rc_circ)

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        # print(paulis)

        self.assertTrue(paulis == [['Gc6'], ['Gc0'], ['Gc3'], ['Gc6'], ['Gc9']])

    # def test_u3_assorted(self):
    #     #TODO: implement

    #CNOT
    def test_cnot_x_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcnot', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([0,0,2,2]) # all X errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcnot':
                    pauli_layer.append('Gcnot')
                    continue
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc3', 'Gc3'], ['Gcnot']])
        self.assertTrue(np.array_equal(pauli_frame, np.array([0,0,2,0]))) # X I is propagated through

    def test_cnot_z_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcnot', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,0,0]) # all Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcnot':
                    pauli_layer.append('Gcnot')
                    continue
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc9', 'Gc9'], ['Gcnot']])
        self.assertTrue(np.array_equal(pauli_frame, np.array([0,2,0,0]))) # Z I is propagated through


    def test_cnot_x_and_z_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcnot', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,2,2]) # all X and Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcnot':
                    pauli_layer.append('Gcnot')
                    continue
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc6', 'Gc6'], ['Gcnot']]) 
        self.assertTrue(np.array_equal(pauli_frame, np.array([0,2,2,0]))) # X Z is propagated through


    def test_cphase_x_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcphase', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([0,0,2,2]) # all X errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcphase':
                    pauli_layer.append('Gcphase')
                    continue
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc3', 'Gc3'], ['Gcphase']]) 
        self.assertTrue(np.array_equal(pauli_frame, np.array([2,2,2,2]))) # XZ ZX is propagated through

    def test_cphase_z_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcphase', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,0,0]) # all Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            # print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcphase':
                    pauli_layer.append('Gcphase')
                    continue
                # print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc9', 'Gc9'], ['Gcphase']]) 
        self.assertTrue(np.array_equal(pauli_frame, np.array([2,2,0,0]))) # Z Z is propagated through


    def test_cphase_x_and_z_errors(self):

        line_labels = ['Q1', 'Q2']
        layers = [[L('Gcphase', ['Q1', 'Q2'], args=None)]]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,2,2,2]) # all X and Z errors

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcphase':
                    pauli_layer.append('Gcphase')
                    continue
                print(gate)
                unitary = U(gate.args)
                clifford = get_clifford_from_unitary(unitary)
                pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        self.assertTrue(paulis == [['Gc6', 'Gc6'], ['Gcphase']])
        self.assertTrue(np.array_equal(pauli_frame, np.array([0,0,2,2]))) # X Z is propagated through


    def test_big_circuit(self):
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [np.pi,0,np.pi]
        Y_args = [np.pi,np.pi/2,np.pi/2]
        Z_args = [0,0,np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                  ]
                
        circ = C(layers, line_labels=line_labels)

        test_layer = np.array([2,0,0,2,2,2,2,2,2,0]) #YXXYZ

        rc = RandomCompilation(rc_strategy='central_pauli', testing=True)

        [rc_circ, _, pauli_frame] = rc.compile(circ, test_layer)

        # print(rc_circ)

        # Output should be

        paulis = []

        U = Gu3()

        # Gc0 is I
        # Gc3 is X
        # Gc6 is Y
        # Gc9 is Z

        for i in range(len(rc_circ)):
            layer = rc_circ.layer_label(i).components
            print(layer)
            pauli_layer = []
            for gate in layer:
                if gate.name == 'Gcphase' or gate.name == 'Gcnot':
                    pauli_layer.append(gate.name)
                else:
                    print(gate)
                    unitary = U(gate.args)
                    clifford = get_clifford_from_unitary(unitary)
                    pauli_layer.append(clifford)
            paulis.append(pauli_layer)
            
        print(paulis)

        print(pauli_frame)

        correct_output_circ = [['Gc6', 'Gc3', 'Gc3', 'Gc6', 'Gc9'],
                               ['Gc9', 'Gc6', 'Gc3', 'Gc0', 'Gc0'],
                               ['Gcnot', 'Gcnot'],
                               ['Gc6', 'Gc9', 'Gc3', 'Gc0', 'Gc3'],
                               ['Gcnot', 'Gcnot'],
                               ['Gcphase', 'Gcphase'],
                               ['Gcnot', 'Gcnot'],
                               ['Gc3', 'Gc6', 'Gc0', 'Gc0', 'Gc6'],
                               ['Gc3', 'Gc6', 'Gc0', 'Gc6', 'Gc6'],
                               ['Gc6', 'Gc3', 'Gc9', 'Gc6', 'Gc0'],
                               ['Gcphase', 'Gcphase']]

        self.assertTrue(paulis == correct_output_circ)
        self.assertTrue(np.array_equal(pauli_frame, np.array([2,0,0,2,0,2,0,0,2,0])))